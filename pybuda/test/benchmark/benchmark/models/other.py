# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Catch-all for random perf testing
"""

import os
import pybuda
import torch

from ..common import benchmark_model
from pybuda.op.common import PyBudaOp
from pybuda.utils import align_up_tile, round_up_div
from pybuda.op.eval.sparse_utils import interleave_tiles, vslice, calculate_conv2d_output_dimensions, create_conv2d_sparse_picker_matrix, conv2d_padding_to_canonical
from pybuda.op.nn import Conv2dModule
from pybuda.config import _get_global_compiler_config


class ConvTModule(pybuda.PyBudaModule):
    """
    ConvTModule
    """

    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        depthwise=False,
        bias=True,
    ):
        super().__init__(name)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = conv2d_padding_to_canonical(padding, tuple(kernel_size))
        self.dilation = dilation

        self.conv = Conv2dModule(
            "conv2d_t",
            in_channels,
            out_channels if not depthwise else in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels if depthwise else 1,
            bias=bias,
        )

    def forward(self, act):
        outy, outx = calculate_conv2d_output_dimensions(act.shape[-2], act.shape[-1], self.kernel_size, self.stride, self.padding, self.dilation)

        y = self.conv(act)
        y = pybuda.op.Reshape("", y, (1, 1, self.out_channels, outy * outx))
        y = pybuda.op.Transpose("", y, 2, 3)

        return y


class SimpleAddModule(pybuda.PyBudaModule):
    """
    Simple add module
    """

    shape = (1, 1, 32, 32)

    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(*self.shape, requires_grad=True)
        self.set_parameter("weights", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):
        return pybuda.op.Add("add1", x, self.weights)


@benchmark_model(configs=["224"])
def big_conv(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    if config == "224":
        input_size = (224, 224)
        cin = 3
        cout = 64
        kH = 7
        kW = 7
        stride = 2
        padding = 3
        dilation = 1
    else:
        raise RuntimeError(f"Invalid config: {config}")

    if microbatch == 0:
        microbatch = 1

    mod = ConvTModule(
        name="big_conv_benchmark",
        in_channels=cin,
        out_channels=cout,
        kernel_size=[kH, kW],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
        depthwise=False,
        bias=False)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    models = {"tt": mod}
    inputs = [torch.rand(microbatch, cin, input_size[0], input_size[1])]
    targets = []

    # if training:
    #     assert False
    #     # models["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return models, inputs, targets, {}


@benchmark_model(configs=["default", "default1", "default2"])
def simple_add(training: bool, config: str, microbatch: int, devtype: str, arch: str):
    mod = SimpleAddModule("simple_add_module")

    models = {"tt": mod}
    inputs = [torch.rand(1, 1, 32, 32)]
    targets = []

    return models, inputs, targets, {}