# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

import pybuda
from pybuda import (
    Tensor,
    Parameter,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
)
from .common import run, compile
import os

@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("factor", [2, 3, 4])
def test_fracture(test_kind, test_device, dim, factor):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def simple_fracture(x, param=None):
        return pybuda.op.Matmul("mm", x, param)

    x = Tensor.create_from_torch(torch.rand((1, 1, 96, 384)))
    param = Parameter(torch.rand((1, 1, 384, 768)), name="m0")

    pybuda.config.insert_fracture_group([("m0", dim, factor)])

    simple_fracture(x, param=param)


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("factor", [2, 3, 4])
def test_fracture_multichip(test_kind, test_device, dim, factor):
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0 or test_device.arch == pybuda.BackendDevice.Blackhole:
        pytest.skip("Skip until #736 is solved")
        
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()

    shape = (1, 1, 64, 64)
    @compile(
        chip_ids=list(range(factor)),
        devtype=test_device.devtype,
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            arch=test_device.arch,
        )
    )
    def simple_fracture(x, param=None):
        return pybuda.op.Matmul("mm", x, param)

    x = Tensor.create_from_torch(torch.rand((1, 1, 96, 384)))
    param = Parameter(torch.rand((1, 1, 384, 768)), name="m0")

    pybuda.config.insert_fracture_group([("m0", dim, factor), "mm"], chip_ids=list(range(factor)))

    compilation_results = simple_fracture(x, param=param)
    placer_solution = compilation_results.pass_specific_output_kwargs["placer_solution"]

    for fracture_idx in range(factor):
        assert placer_solution.chip_id(f"fractured_{fracture_idx}_mm") == fracture_idx


def test_fracture_2d(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def simple_fracture_2d(x, Win=None):
        return pybuda.op.Matmul("e0", x, Win)

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 384)))
    Win = Parameter(torch.rand((1, 1, 384, 512)), name="Win")

    pybuda.config.insert_fracture_group([("Win", [-2, -1], [3, 2]), "e0"])

    simple_fracture_2d(x, Win=Win)


# Figure 2a from paper https://arxiv.org/pdf/2211.05102.pdf
def test_fracture_1d_weight_stationary(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_1d_weight_stationary(x, Win=None, Wout=None):
        x = pybuda.op.Matmul("e0", x, Win)
        x = pybuda.op.Gelu("gelu", x)
        x = pybuda.op.Matmul("e1", x, Wout)
        return x

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 384)))
    Win = Parameter(torch.rand((1, 1, 384, 512)), name="Win")
    Wout = Parameter(torch.rand((1, 1, 512, 384)), name="Wout")

    pybuda.config.insert_fracture_group([("Win", -1, 2), ("Wout", -2, 2)])

    fracture_1d_weight_stationary(x, Win=Win, Wout=Wout)


# Figure 2b from paper https://arxiv.org/pdf/2211.05102.pdf
def test_fracture_2d_weight_stationary(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_2d_weight_stationary(x, Win=None, Wout=None):
        x = pybuda.op.Matmul("e0", x, Win)
        x = pybuda.op.Gelu("gelu", x)
        x = pybuda.op.Matmul("e1", x, Wout)
        return x

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 384)))
    Win = Parameter(torch.rand((1, 1, 384, 512)), name="Win")
    Wout = Parameter(torch.rand((1, 1, 512, 384)), name="Wout")

    pybuda.config.insert_fracture_group([("Win", [-2, -1], [2, 4]), ("Wout", [-2, -1], [4, 3])])

    fracture_2d_weight_stationary(x, Win=Win, Wout=Wout)


def test_fracture_slice(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_slice(x, Win=None):
        Win = pybuda.op.HSlice("", Win, 12)
        x = pybuda.op.Matmul("e0", x, Win)
        x = pybuda.op.Gelu("gelu", x)
        return x

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 384)))
    Win = Parameter(torch.rand((1, 1, 384, 128*12)), name="Win")

    pybuda.config.insert_fracture_group([("e0", -3, 12), "gelu"])

    fracture_slice(x, Win=Win)

@pytest.mark.parametrize("config", [
    (-1, 2),
    (-2, 4),
    (-3, 12),
    ([-1, -2], [2, 2]),
    ([-2, -1], [2, 2]),
    ([-1, -3], [2, 6]),
    ([-3, -1], [3, 2]),
    ([-2, -3], [2, 2]),
    ([-3, -2], [2, 2]),
    ([-3, -2, -1], [2, 2, 2]),
    ([-2, -1, -3], [2, 2, 3]),
    ([-1, -3, -2], [4, 2, 3]),
    ([-1, -2, -3], [2, 2, 2]),
    ([-2, -3, -1], [3, 6, 2]),
    ([-3, -1, -2], [3, 4, 2]),
])
def test_fracture_bcast(test_kind, test_device, config):
    pytest.skip("tenstorrent/pybuda#1903")
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_bcast(x, Win=None, bias=None):
        x = pybuda.op.Matmul("e0", x, Win)
        x = pybuda.op.Add("add", x, bias)
        x = pybuda.op.Gelu("gelu", x)
        return x

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 384)))
    Win = Parameter(torch.rand((1, 12, 384, 128)), name="Win")
    bias = Tensor.create_from_torch(torch.rand(1), constant=True)

    pybuda.config.insert_fracture_group([("Win",) + config, "gelu"])

    fracture_bcast(x, Win=Win, bias=bias)


def test_fracture_output(test_kind, test_device):
    os.environ["PYBUDA_CONCAT_ON_HOST"] = "1"

    if test_kind.is_training():
        pytest.skip()

    # TODO: Runtime transform eval needed, skip verification for now
    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_output(x, Win=None):
        x = pybuda.op.Matmul("mm", x, Win)
        return x

    x = Tensor.create_from_torch(torch.rand(1, 128, 128))
    Win = Parameter(torch.rand(128, 1024), name="Win")
    pybuda.config.insert_fracture_group([("mm", -1, 2)])
    fracture_output(x, Win=Win)


def test_fracture_fork_join(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_fork_join(x, y, Win=None):
        x = pybuda.op.Matmul("e0", x, Win)
        h = pybuda.op.HSlice("", x, 16)
        m = pybuda.op.Multiply("multiply", h, y)
        f0 = pybuda.op.Gelu("gelu0", m)
        f1 = pybuda.op.Gelu("gelu1", h)
        j = pybuda.op.Multiply("join", f0, f1)
        return j

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 384)))
    y = Tensor.create_from_torch(torch.rand((1, 16, 128, 128)))
    Win = Parameter(torch.rand((1, 1, 384, 128*16)), name="Win")

    pybuda.config.insert_fracture_group([("e0", -2, 2), ("join", -2, 2)])

    fracture_fork_join(x, y, Win=Win)


def test_fracture_fork_input(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def fracture_fork_input(x, Win=None):
        f0 = pybuda.op.Gelu("gelu0", x)
        f1 = pybuda.op.Gelu("gelu1", x)
        return f0, f1

    x = Tensor.create_from_torch(torch.rand((1, 1, 128, 128)))

    pybuda.config.insert_fracture_group([("gelu0", -2, 2)])
    pybuda.config.insert_fracture_group([("gelu1", -1, 2)])

    fracture_fork_input(x)


@pytest.mark.parametrize("factor", [[2, 2], [2, 4], [4, 2], [2, 6], [6, 2]])
@pytest.mark.parametrize("fork", [False, True])
def test_mixed_factors(test_kind, test_device, factor, fork):
    if test_kind.is_training():
        pytest.skip()

    @run(
        VerifyConfig(
            test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch
        )
    )
    def mixed_factors(x, Win=None):
        if fork:
            f0 = pybuda.op.Gelu("gelu0", x)
            f1 = pybuda.op.Gelu("gelu1", x)
            return f0, f1
        else:
            f0 = pybuda.op.Gelu("gelu0", x)
            return pybuda.op.Gelu("gelu1", f0)

    x = Tensor.create_from_torch(torch.rand((1, 1, 384, 384)))

    compile_cfg = pybuda.config._get_global_compiler_config()
    compile_cfg.scheduler_policy = "Topological"
    pybuda.config.insert_fracture_group([("gelu0", -1, factor[0]), ("gelu1", -1, factor[1])])

    mixed_factors(x)


def test_fracture_transpose(test_device):

    @run(test_device)
    def fracture_transpose(x, y=None):
        f0 = pybuda.op.Gelu("gelu0", x)
        f0 = pybuda.op.Transpose("t0", f0, -2, -1)
        return pybuda.op.Add("add0", f0, y)

    x = Tensor.create_from_torch(torch.rand((1, 1, 384, 384)))
    y = Parameter(torch.rand((1, 1, 384, 384)), name="y")

    pybuda.config.insert_fracture_group([("gelu0", -1, 2), "add0"])

    fracture_transpose(x, y=y)
