# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test build-in loss
"""

import pybuda

from pybuda.op.loss import L1Loss, CrossEntropyLoss
from pybuda.verify import VerifyConfig, verify_module
import os

class BudaTest(pybuda.PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (64, 64)

    def __init__(self, name: str):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        add = pybuda.op.Add("add", m1, m2)
        return add


def test_l1_loss(test_kind, test_device):
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    verify_module(BudaTest("test_l1_loss"), [(1, *BudaTest.shape), (1, *BudaTest.shape)],
            loss_module=L1Loss("l1_loss"),
            verify_cfg=VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch))

def test_ce_loss(test_kind, test_device):
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    verify_module(BudaTest("test_ce_loss"), [(1, *BudaTest.shape), (1, *BudaTest.shape)],
            loss_module=CrossEntropyLoss("ce_loss"),
            verify_cfg=VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch))
