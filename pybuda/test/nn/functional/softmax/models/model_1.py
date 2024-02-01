# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Softmax operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch
from torch.distributions import Uniform, Normal

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaSoftmaxTest(PyBudaModule):
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
        dim, 
        stable):
        super().__init__("Buda Test 1")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.testname = "Operator softmax Test 1"
        self.shape = shape
        self.dim = dim
        self.stable = stable
        
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        input = BudaSoftmaxTest.INPUTS_DISTRIBUTION(
            BudaSoftmaxTest.INPUTS_RANGE_MIN, 
            BudaSoftmaxTest.INPUTS_RANGE_MAX).sample(self.shape)
        self.inputs = [Tensor.create_from_torch(input)]

        weights = BudaSoftmaxTest.WEIGHTS_DISTRIBUTION(
            BudaSoftmaxTest.WEIGHTS_RANGE_MIN, 
            BudaSoftmaxTest.WEIGHTS_RANGE_MAX).sample(self.shape)
        weights.requires_grad = True
        self.set_parameter("train_param", weights)

    def forward(self, x):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x, self.train_param)

        # Layer 3
        sm1 = nn.Softmax("sm1", x, dim=self.dim, stable=self.stable)
        sm2 = nn.Softmax("sm2", mul1, dim=self.dim, stable=self.stable)
        sm3 = nn.Softmax("sm3", self.train_param, dim=self.dim, stable=self.stable)

        # Layer 4
        add1 = pybuda.op.Add("add1", sm1, sm2)
        mul2 = pybuda.op.Multiply("mul2", sm2, sm3)

        # Layer 5
        mul3 = pybuda.op.Multiply("mul3", add1, mul2)
        sm4 = nn.Softmax("sm4", mul3, dim=self.dim, stable=self.stable)

        return sm4

    def values(self):
        return [item.value() for item in self.inputs]