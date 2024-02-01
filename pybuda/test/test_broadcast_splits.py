# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Bert-related tests for the new front-end
#

import pytest
import torch

import pybuda
from pybuda import (
    TTDevice,
    BackendType,
    Tensor,
    Parameter,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
    SGD,
    PyBudaModule,
)


class BroadcastSplitModule(PyBudaModule):
    """
    BroadcastSplitModule
    """

    def __init__(self, name):
        super().__init__(name)

    def forward(self, op1, op2, op3):
        bc1 = pybuda.op.Broadcast("bc1", op2, 3, 32)
        bc2 = pybuda.op.Broadcast("bc2", bc1, 2, 32)
        # implicit Z broadcast in add
        add1 = pybuda.op.Add("add1", op1, bc2)
        mm1 = pybuda.op.Matmul("mm1", add1, op3)
        return mm1


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_broadcast_split(mode):
    training = mode == "training"

    if training:
        pytest.skip()  # skip for now

    op1 = Tensor.create_from_torch(torch.rand(1, 2, 32, 32, requires_grad=training))
    op2 = Tensor.create_from_torch(torch.rand(1, 1, 1, 1, requires_grad=training))
    op3 = Tensor.create_from_torch(torch.rand(1, 2, 32, 1, requires_grad=training))

    mod = BroadcastSplitModule("broadcast_split")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "broadcast_split",
        op1,
        op2,
        op3,
        compiler_cfg=CompilerConfig(enable_training=training, enable_broadcast_splitting=True),
        verify_cfg=VerifyConfig(run_golden=False),
    )
