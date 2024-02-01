# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# NN modules
#

import pytest
import torch

import pybuda
import pybuda.op
import pybuda.op.nn as nn
from pybuda import (
    PyBudaModule,
    TTDevice,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
)
from pybuda._C.backend_api import BackendType

verify_cfg = VerifyConfig(run_golden=True) # run backend golden on each test

class SoftmaxTest(PyBudaModule):
    """
    Test wrapper for softmax
    """

    def __init__(self, name, stable, dim):
        super().__init__(name)
        self.dim = dim
        self.stable = stable

    def forward(self, act):
        return nn.Softmax("softmax", act, dim=self.dim, stable=self.stable)


class LayernormTest(PyBudaModule):
    """
    Test wrapper for layernorm
    """

    def __init__(self, name, wshape, bshape):
        super().__init__(name)
        self.weights = pybuda.Parameter(*wshape, requires_grad=True)
        self.bias = pybuda.Parameter(*bshape, requires_grad=True)

    def forward(self, act):
        return nn.Layernorm("layernorm", act, self.weights, self.bias)


@pytest.mark.parametrize("shape", ([1, 1, 64, 64], [128, 128], [128, 768], [512, 256]), ids=["shape1x1x64x64", "shape128x128", "shape128x768", "shape512x256"])
@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize("stable", (True, False), ids=["StableSoftmax", "OriginalSoftmax"])
@pytest.mark.parametrize("dim", (-1, ))
def test_softmax(
    test_device,
    training, 
    recompute, 
    stable, 
    dim, 
    shape
):

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    mod = SoftmaxTest("test_module", stable, dim)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(mod)

    from torch.distributions import Uniform, Normal
    act1 = Normal(-1.0, 1.0).sample(shape)
    act1.requires_grad = True
    act1 = Tensor.create_from_torch(act1)

    pybuda_compile(
        tt0,
        "softmax",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=verify_cfg,
    )


@pytest.mark.parametrize("shape, wshape, bshape", (([1, 1, 17, 64], [1, 1, 1, 64], [1, 1, 1, 64]), ([17, 64], [1, 64], [1, 64])))
@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_layernorm(
    training, 
    recompute,
    shape,
    wshape,
    bshape
):

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    mod = LayernormTest("test_module", wshape, bshape)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(
        torch.rand(*shape, requires_grad=True)
    )

    mod.set_parameter("weights", torch.rand(*wshape, requires_grad=True))
    mod.set_parameter("bias", torch.rand(*bshape, requires_grad=True))

    pybuda_compile(
        tt0,
        "layernorm",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=verify_cfg,
    )
