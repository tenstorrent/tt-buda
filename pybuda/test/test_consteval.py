# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch

import pybuda
from pybuda import (
    Tensor,
    Parameter,
    CompilerConfig,
    VerifyConfig,
)
from .common import run

optimizer = {"type": "sgd", "params": {"learning_rate": 50.0}}


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 1, 32, 64), (1, 1, 64, 32)),
        ((1, 1, 32, 64), (1, 2, 32, 32)),
    ],
)
def test_consteval_simple(test_kind, test_device, shapes):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            optimizer=optimizer,
        ),
    )
    def consteval_simple(x, param=None):
        param = pybuda.op.Reshape("reshape0", param, shapes[0])
        return pybuda.op.Multiply("mul0", x, param)

    x = Tensor.create_from_torch(
        torch.rand(*shapes[0], requires_grad=test_kind.is_training())
    )
    param = Parameter.create_from_torch(
        torch.rand(*shapes[1], requires_grad=test_kind.is_training())
    )
    consteval_simple(x, param=param)


def test_consteval_param_chain(test_kind, test_device):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            optimizer=optimizer,
        ),
    )
    def consteval_param_chain(x, param=None):
        param = pybuda.op.Reshape("reshape0", param, (1, 10, 8, 8))
        param = pybuda.op.Transpose("transpose0", param, 1, 3)
        param = pybuda.op.Transpose("transpose1", param, 2, 3)
        param = pybuda.op.Reshape("reshape1", param, (1, 4, 16, 10))
        param = pybuda.op.Transpose("transpose2", param, 2, 3)
        param = pybuda.op.Reshape("reshape2", param, (1, 2, 10, 32))
        return pybuda.op.Multiply("mul0", x, param)

    x = Tensor.create_from_torch(
        torch.rand((1, 2, 10, 32), requires_grad=test_kind.is_training())
    )
    c = Parameter.create_from_torch(
        torch.rand((1, 20, 4, 8), requires_grad=test_kind.is_training())
    )
    consteval_param_chain(x, param=c)


def test_consteval_partial(test_kind, test_device):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            optimizer=optimizer,
        ),
    )
    def consteval_partial(x, param=None):
        param = pybuda.op.Transpose("transpose", param, 2, 3)
        param = pybuda.op.Exp("exp", param)
        return pybuda.op.Multiply("mul0", x, param)

    x = Tensor.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    c = Parameter.create_from_torch(
        torch.rand((1, 1, 32, 256), requires_grad=test_kind.is_training())
    )
    consteval_partial(x, param=c)


def test_consteval_fork(test_kind, test_device):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            optimizer=optimizer,
        ),
    )
    def consteval_fork(x, const=None):
        a = pybuda.op.Transpose("transpose", const, 2, 3)
        b = pybuda.op.Exp("exp", a)
        c = pybuda.op.Log("log", a)
        d = pybuda.op.Multiply("mul0", x, b)
        e = pybuda.op.Multiply("mul1", d, c)
        return e

    x = Tensor.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    c = Tensor.create_from_torch(torch.rand((1, 1, 32, 256)), constant=True)
    consteval_fork(x, const=c)


def test_consteval_binary(test_kind, test_device):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            optimizer=optimizer,
        ),
    )
    def consteval_binary(x, a=None, b=None):
        a = pybuda.op.Transpose("ta", a, 2, 3)
        a = pybuda.op.Exp("expa", a)
        b = pybuda.op.Transpose("tb", b, 2, 3)
        b = pybuda.op.Log("logb", b)
        c = pybuda.op.Multiply("mulc", a, b)
        c = pybuda.op.Transpose("tc", c, 2, 3)
        return pybuda.op.Multiply("mul0", x, c)

    x = Tensor.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    a = Parameter.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    b = Parameter.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    consteval_binary(x, a=a, b=b)


def test_consteval_binary_fork(test_kind, test_device):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            optimizer=optimizer,
        ),
    )
    def consteval_binary_fork(x, a=None, b=None):
        a = pybuda.op.Transpose("ta", a, 2, 3)
        a = pybuda.op.Exp("expa", a)
        b = pybuda.op.Transpose("tb", b, 2, 3)
        b = pybuda.op.Log("logb", b)
        c = pybuda.op.Multiply("mulc", a, b)
        d = pybuda.op.Transpose("tc0", c, 2, 3)
        e = pybuda.op.Transpose("tc1", c, 2, 3)
        e = pybuda.op.Exp("exp0", e)
        f = pybuda.op.Multiply("mul0", x, d)
        g = pybuda.op.Multiply("mul1", x, e)
        return pybuda.op.Multiply("mul2", f, g)

    x = Tensor.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    a = Parameter.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    b = Parameter.create_from_torch(
        torch.rand((1, 1, 256, 32), requires_grad=test_kind.is_training())
    )
    consteval_binary_fork(x, a=a, b=b)
