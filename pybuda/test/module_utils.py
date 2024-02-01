# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Container for various modules used for testing
# 

import pybuda
from pybuda import PyBudaModule
from pybuda.op.nn import Conv2dModule
from pybuda.op.eval.sparse_utils import calculate_conv2d_output_dimensions, conv2d_padding_to_canonical


class Conv2dTModule(PyBudaModule):
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
        add_reshape_transpose_to_end=True,
    ):
        super().__init__(name)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = conv2d_padding_to_canonical(padding, self.kernel_size)
        self.dilation = dilation

        self.bias = bias

        self.add_reshape_transpose_to_end = add_reshape_transpose_to_end

        self.conv = Conv2dModule(
            name="conv2d_t",
            in_channels=self.in_channels,
            out_channels=self.out_channels if not depthwise else in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.in_channels if depthwise else 1,
            bias=self.bias,
        )

    def forward(self, act):
        outy, outx = calculate_conv2d_output_dimensions(act.shape[-2], act.shape[-1], self.kernel_size, self.stride, self.padding, self.dilation)

        y = self.conv(act)

        if self.add_reshape_transpose_to_end:
            y = pybuda.op.Reshape("", y, (1, 1, self.out_channels, outy * outx))
            y = pybuda.op.Transpose("", y, 2, 3)

        return y
