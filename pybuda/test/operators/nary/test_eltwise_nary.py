# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise nary operators
#

import pybuda.tensor
import pytest

import torch
import os

import pybuda
import pybuda.op
from pybuda import PyBudaModule, Tensor, VerifyConfig
from test.common import run
from pybuda.verify import TestKind, verify_module

verify_cfg = VerifyConfig(
    run_golden=True, run_net2pipe=True
)  # Run backend golden check on all tests in here


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 10, 32, 32),
        (1, 32, 16, 16),
    ],
)
@pytest.mark.parametrize("axis", [-3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("num_operands", [2, 3])
def test_interleave(test_kind, test_device, input_shape, axis, stride, num_operands):
    class Model(PyBudaModule):
        def __init__(self, name, axis, stride):
            super().__init__(name)
            self.axis = axis
            self.stride = stride

        def forward(self, *operands):
            x = pybuda.op.Interleave(
                "interleave0", *operands, axis=self.axis, stride=self.stride
            )
            return x

    input_shapes = tuple([input_shape for _ in range(num_operands)])
    mod = Model("interleave_test", axis, stride)
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )


@pytest.mark.parametrize("dim", [1, 2, -1])
@pytest.mark.parametrize("aligned", [True, False])
def test_concat(test_kind, test_device, dim, aligned):
    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        ),
    )
    def simple_concat(a, b):
        return pybuda.op.Concatenate("", a, b, axis=dim)

    if aligned:
        shapes = {
            -1: (1, 3, 128, 96),
            2: (1, 3, 1024, 32),
            1: (1, 1, 128, 32),
        }
        a = Tensor.create_from_torch(
            torch.randn((1, 3, 128, 32), requires_grad=test_kind.is_training())
        )
    else:
        shapes = {
            -1: (1, 3, 128, 6),
            2: (1, 3, 128, 6),
            1: (1, 1, 128, 6),
        }
        a = Tensor.create_from_torch(
            torch.randn((1, 3, 128, 6), requires_grad=test_kind.is_training())
        )
    b = Tensor.create_from_torch(
        torch.randn(shapes[dim], requires_grad=test_kind.is_training())
    )
    c = simple_concat(a, b)


def test_concat_two_kinds_pad(test_device):
    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            self.add_parameter(
                "w", pybuda.Parameter(*(1, 1, 352, 192), requires_grad=True)
            )

        def forward(self, in0, in1, in2, in3, in4, in5, y):
            in0 = pybuda.op.Multiply("m0", in0, in0)
            in1 = pybuda.op.Multiply("m1", in1, in2)
            in2 = pybuda.op.Multiply("m2", in2, in3)
            in3 = pybuda.op.Multiply("m3", in3, in4)
            in4 = pybuda.op.Multiply("m4", in4, in4)
            in5 = pybuda.op.Multiply("m5", in5, in1)
            x = pybuda.op.Concatenate("", in0, in1, in2, in3, in4, in5, axis=-1)
            x = pybuda.op.Multiply("m6", x, y)
            x = pybuda.op.PadTile("p0", x, -1, 336)
            x = pybuda.op.Matmul("mm0", x, self.get_parameter("w"))
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config() # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    # compiler_cfg.place_on_new_epoch("m6_transpose_nop_0")
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    # input shape
    common_len = 3136
    input_shapes = (
        (1, 1, common_len, 96),
        (1, 1, common_len, 48),
        (1, 1, common_len, 48),
        (1, 1, common_len, 48),
        (1, 1, common_len, 48),
        (1, 1, common_len, 48),
        (1, 1, common_len, 336),
    )
    mod = Module("test_concat_two_kinds_pad")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )

    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{}"



