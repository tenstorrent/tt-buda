# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import pytest
from pybuda.verify import verify_module, VerifyConfig, TestKind

class FusingStreamLimitsStress(pybuda.PyBudaModule):
    """
    Module which tests recompile when fused op doesn't satisfy stream contraints.
    """

    shape = (1, 1, 3200, 128)

    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):
        matmuls = []
        for i in range(10):
            matmuls.append(pybuda.op.Matmul(f"matmul_{i}", act1, self.weights))

        for i in range(10):
            matmuls.append(pybuda.op.Matmul(f"matmul_{i+10}", act2, self.weights))

        # Expecting fusing of ops below
        add = pybuda.op.Add("", matmuls[0], matmuls[1])
        for i in range(2, 20):
            add = pybuda.op.Add("", add, matmuls[i])

        return add

def test_recompile_fuse_stream_limits(test_device):
    pytest.skip()

    # Setting target cycles to 0 causes us to hit stream constraints on fused op.
    os.environ["PYBUDA_RIBBON_TARGET_CYCLES"] = "0"
    os.environ["PYBUDA_TEMP_BALANCER_MODEL_PCIE_BW"] = "0"
    os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"

    # Enable recompilation to recover from net2pipe failure.
    os.environ["PYBUDA_AUTO_RECOMPILE"] = "1"

    run_net2pipe = not test_device.is_silicon()

    pybuda.config.set_configuration_options(balancer_policy="Ribbon")
    verify_module(FusingStreamLimitsStress("recompile_fuse_stream_limits"), [FusingStreamLimitsStress.shape, FusingStreamLimitsStress.shape],
            VerifyConfig(test_kind=TestKind.INFERENCE, arch=test_device.arch, devtype=test_device.devtype, run_net2pipe=run_net2pipe))

