# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   LeakyRelu operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch
from torch.distributions import Normal

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaLeakyReluTest(PyBudaModule):
    """
        Buda Test 1

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
        alpha
    ):
        super().__init__("Buda Test 1")

        self.testname = "Operator LeakyRelu, Test 1"
        self.shape = shape
        self.alpha = alpha
        
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        input = BudaLeakyReluTest.INPUTS_DISTRIBUTION(
            BudaLeakyReluTest.INPUTS_RANGE_MIN, 
            BudaLeakyReluTest.INPUTS_RANGE_MAX).sample(self.shape)
        self.inputs = [Tensor.create_from_torch(input)]

        weights = BudaLeakyReluTest.WEIGHTS_DISTRIBUTION(
            BudaLeakyReluTest.WEIGHTS_RANGE_MIN, 
            BudaLeakyReluTest.WEIGHTS_RANGE_MAX).sample(self.shape)
        weights.requires_grad = True
        self.set_parameter("train_param", weights)

    def forward(self, x):

        # Layer 2
        mul = pybuda.op.Multiply("mul", x, self.train_param)

        # Layer 3
        lrelu = pybuda.op.LeakyRelu("lrelu", mul, alpha=self.alpha)

        return lrelu
