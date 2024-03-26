# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Versim-related tests for end-to-end simulation
#
from pybuda import pybuda
from pybuda._C.backend_api import BackendDevice, BackendType
from pybuda.tensor import Tensor
from pybuda.verify.config import TestKind, VerifyConfig
import pytest
import torch
from test.common import run

def test_versim_simple_add(test_device):
    # Run only versim tests
    if test_device.devtype != BackendType.Versim:
        pytest.skip()

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_add(a, b):
        c = pybuda.op.Add("add0", a, b)
        return c

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.input_queues_on_host = False
    compiler_cfg.output_queues_on_host = False
    compiler_cfg.balancer_op_override("add0", "grid_shape", (1,1))

    shape = (1, 3, 128, 128)
    a = Tensor.create_from_torch(torch.randn(shape))
    b = Tensor.create_from_torch(torch.randn(shape))
    simple_add(a, b)