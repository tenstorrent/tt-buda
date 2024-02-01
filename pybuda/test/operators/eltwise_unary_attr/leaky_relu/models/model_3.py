# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
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
        Buda Test 3

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
        super().__init__("Buda Test 3")

        self.testname = "Operator LeakyRelu, Test 3"
        self.shape = shape
        self.alpha = alpha
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(2):
            input = BudaLeakyReluTest.INPUTS_DISTRIBUTION(
                BudaLeakyReluTest.INPUTS_RANGE_MIN, 
                BudaLeakyReluTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 3):
            weights = BudaLeakyReluTest.WEIGHTS_DISTRIBUTION(
                BudaLeakyReluTest.WEIGHTS_RANGE_MIN, 
                BudaLeakyReluTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)


    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param1)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        lrelu1 = pybuda.op.LeakyRelu("lrelu1", mul1, alpha=self.alpha)
        lrelu2 = pybuda.op.LeakyRelu("lrelu2", mul2, alpha=self.alpha)
        lrelu3 = pybuda.op.LeakyRelu("lrelu3", mul3, alpha=self.alpha)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", lrelu1, self.train_param1)
        mul5 = pybuda.op.Multiply("mul5", lrelu3, self.train_param2)
        add1 = pybuda.op.Add("add1", lrelu2, x2)

        # Layer 5
        lrelu4 = pybuda.op.LeakyRelu("lrelu4", mul4, alpha=self.alpha)
        lrelu5 = pybuda.op.LeakyRelu("lrelu5", add1, alpha=self.alpha)
        lrelu6 = pybuda.op.LeakyRelu("lrelu6", mul5, alpha=self.alpha)

        # Layer 6
        mul6 = pybuda.op.Multiply("mul6", lrelu4, lrelu2)
        mul7 = pybuda.op.Multiply("mul7", mul2, lrelu6)
        add2 = pybuda.op.Add("add2", lrelu5, mul3)

        # Layer 7
        lrelu7 = pybuda.op.LeakyRelu("lrelu7", mul6, alpha=self.alpha)
        lrelu8 = pybuda.op.LeakyRelu("lrelu8", add2, alpha=self.alpha)
        lrelu9 = pybuda.op.LeakyRelu("lrelu9", mul7, alpha=self.alpha)

        # Layer 8
        mul8 = pybuda.op.Multiply("mul8", lrelu7, mul4)
        mul9 = pybuda.op.Multiply("mul9", lrelu8, lrelu6)
        mul10 = pybuda.op.Multiply("mul10", lrelu9, lrelu3)

        return mul8, mul9, mul10