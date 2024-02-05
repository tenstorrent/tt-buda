# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import pytest
import torch

from pybuda.config import CompileDepth, _get_global_compiler_config
from .common import run


def test_stream_transpose(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        pybuda.VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        ),
    )
    def stream_transpose(a, b, param=None):
        x = pybuda.op.Add("add0", a, b)
        x = pybuda.op.Transpose("transpose0", x, 2, 3)
        x = pybuda.op.Matmul("mm0", x, param)
        return x

    compiler_cfg = pybuda.config._get_global_compiler_config()

    pybuda.config.override_op_size("add0", (1, 1))
    pybuda.config.override_op_size("transpose0", (1, 1))
    pybuda.config.override_op_size("mm0", (1, 1))

    shape = (1, 1, 32, 16384)
    a = pybuda.Tensor.create_from_torch(
        torch.rand(*shape, requires_grad=test_kind.is_training())
    )
    b = pybuda.Tensor.create_from_torch(
        torch.rand(*shape, requires_grad=test_kind.is_training())
    )
    c = pybuda.Tensor.create_from_torch(torch.rand(1, 1, 32, 32), constant=True)
    stream_transpose(a, b, param=c)


def test_stream_to_slice(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        pybuda.VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        ),
    )
    def stream_to_slice(x):
        x = pybuda.op.Buffer("buf0", x)
        x = pybuda.op.VSlice("vslice0", x, 512)
        x = pybuda.op.Buffer("buf1", x)
        return x

    compiler_cfg = pybuda.config._get_global_compiler_config()

    pybuda.config.override_op_size("buf0", (1, 1))
    pybuda.config.override_op_size("buf1", (1, 1))

    shape = (1, 1, 16384, 32)
    a = pybuda.Tensor.create_from_torch(
        torch.rand(*shape, requires_grad=test_kind.is_training())
    )
    stream_to_slice(a)


@pytest.mark.parametrize("mode", ["producer_streaming", "consumer_streaming", "both_streaming"])
def test_stream_slice_transpose(test_kind, test_device, mode):
    if test_kind.is_training():
        pytest.skip()

    @run(
        pybuda.VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        ),
    )
    def stream_slice_transpose(x):
        x = pybuda.op.Buffer("producer", x)
        x = pybuda.op.VSlice("vslice", x, 2)
        x = pybuda.op.Transpose("consumer", x, 2, 3)
        return x

    compiler_cfg = pybuda.config._get_global_compiler_config()

    if mode == "producer_streaming":
        pybuda.config.override_t_stream_shape("producer", (2, 1))
        pybuda.config.override_t_stream_shape("consumer", (1, 1))
    elif mode == "consumer_streaming":
        pybuda.config.override_t_stream_shape("producer", (1, 1))
        pybuda.config.override_t_stream_shape("consumer", (1, 2))
    elif mode == "both_streaming":
        pybuda.config.override_t_stream_shape("producer", (2, 1))
        pybuda.config.override_t_stream_shape("consumer", (1, 2))

    shape = (1, 1, 128, 32)
    a = pybuda.Tensor.create_from_torch(
        torch.rand(*shape, requires_grad=test_kind.is_training())
    )
    stream_slice_transpose(a)


@pytest.mark.parametrize("dir", ["r", "c"])
def test_stream_interleave(test_device, dir):
    pybuda.config.set_configuration_options(balancer_policy="MaximizeTMinimizeGrid")
    pybuda.config.override_t_stream_dir("interleave", dir)

    @run(test_device)
    def stream_interleave(a, b):
        return pybuda.op.Interleave("interleave", a, b, axis=-3, stride=1)

    shape = (1, 4, 512, 512)
    a = pybuda.Tensor.create_from_torch(torch.randn(*shape))
    b = pybuda.Tensor.create_from_torch(torch.randn(*shape))
    stream_interleave(a, b)


def test_manual_streaming(test_device):

    @run(test_device)
    def manual_stream(x):
        x = pybuda.op.Buffer("buf0", x)
        x = pybuda.op.Buffer("buf1", x)
        x = pybuda.op.Buffer("buf2", x)
        return x

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.manual_t_streaming = True

    pybuda.config.override_t_stream_shape("buf1", (4, 1))

    shape = (1, 1, 128, 128)
    a = pybuda.Tensor.create_from_torch(torch.rand(*shape))
    manual_stream(a)
