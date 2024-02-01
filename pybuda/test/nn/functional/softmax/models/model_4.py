# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
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
        Buda Test 4

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
        super().__init__("Buda Test 4")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.testname = "Operator softmax Test 4"
        self.shape = shape
        self.dim = dim
        self.stable=stable
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = BudaSoftmaxTest.INPUTS_DISTRIBUTION(
                BudaSoftmaxTest.INPUTS_RANGE_MIN, 
                BudaSoftmaxTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = BudaSoftmaxTest.WEIGHTS_DISTRIBUTION(
                BudaSoftmaxTest.WEIGHTS_RANGE_MIN, 
                BudaSoftmaxTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        sm1 = nn.Softmax("sm1", mul1, dim=self.dim, stable=self.stable) 
        sm2 = nn.Softmax("sm2", mul2, dim=self.dim, stable=self.stable)
        sm3 = nn.Softmax("sm3", mul3, dim=self.dim, stable=self.stable)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", sm1, self.train_param1)
        mul5 = pybuda.op.Multiply("mul5", sm2, self.train_param2)
        mul6 = pybuda.op.Multiply("mul6", sm3, self.train_param3)

        # Layer 5
        sm4 = nn.Softmax("sm4", mul4, dim=self.dim, stable=self.stable) 
        sm5 = nn.Softmax("sm5", mul5, dim=self.dim, stable=self.stable)
        sm6 = nn.Softmax("sm6", mul6, dim=self.dim, stable=self.stable)

        # Layer 6
        add1 = pybuda.op.Add("add1", sm4, self.train_param1)
        add2 = pybuda.op.Add("add2", sm5, self.train_param2)
        add3 = pybuda.op.Add("add3", sm6, self.train_param3)

        # Layer 7
        mul7 = pybuda.op.Multiply("mul7", add1, sm2)
        mul8 = pybuda.op.Multiply("mul8", add2, self.train_param2)

        # Layer 8
        sm7 = nn.Softmax("sm7", mul7, dim=self.dim, stable=self.stable)
        sm8 = nn.Softmax("sm8", mul8, dim=self.dim, stable=self.stable)
        sm9 = nn.Softmax("sm9", add3, dim=self.dim, stable=self.stable)

        return sm7, sm8, sm9

    def values(self):
        return [item.value() for item in self.inputs]   