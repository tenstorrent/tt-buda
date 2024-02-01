# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Clip operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch
from torch.distributions import Normal

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaClipTest(PyBudaModule):
    """
        Buda Test 2

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(
        self,
        shape,
        min_value,
        max_value
    ):
        super().__init__("Buda Test 2")

        self.testname = "Operator Clip, Test 2"
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(2):
            input = BudaClipTest.INPUTS_DISTRIBUTION(
                BudaClipTest.INPUTS_RANGE_MIN, 
                BudaClipTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 3):
            weights = BudaClipTest.WEIGHTS_DISTRIBUTION(
                BudaClipTest.WEIGHTS_RANGE_MIN, 
                BudaClipTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)


    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param1)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        clip1 = pybuda.op.Clip("clip1", mul1, min=self.min_value, max=self.max_value)
        clip2 = pybuda.op.Clip("clip2", mul2, min=self.min_value, max=self.max_value)
        clip3 = pybuda.op.Clip("clip3", mul3, min=self.min_value, max=self.max_value)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", clip1, clip2)
        mul5 = pybuda.op.Multiply("mul5", clip2, clip3)

        # Layer 5
        clip4 = pybuda.op.Clip("clip4", mul4, min=self.min_value, max=self.max_value)
        clip5 = pybuda.op.Clip("clip5", mul5, min=self.min_value, max=self.max_value)

        # Layer 6
        mul6 = pybuda.op.Multiply("mul6", clip4, clip5)

        return mul6
