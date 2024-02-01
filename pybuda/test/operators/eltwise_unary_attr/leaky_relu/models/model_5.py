# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
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
        alpha
    ):
        super().__init__("Buda Test 4")

        self.testname = "Operator LeakyRelu, Test 4"
        self.shape = shape
        self.alpha = alpha
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = BudaLeakyReluTest.INPUTS_DISTRIBUTION(
                BudaLeakyReluTest.INPUTS_RANGE_MIN, 
                BudaLeakyReluTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = BudaLeakyReluTest.WEIGHTS_DISTRIBUTION(
                BudaLeakyReluTest.WEIGHTS_RANGE_MIN, 
                BudaLeakyReluTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)



    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        lrelu1 = pybuda.op.LeakyRelu("lrelu1", mul1, alpha=self.alpha)
        lrelu2 = pybuda.op.LeakyRelu("lrelu2", mul2, alpha=self.alpha)
        lrelu3 = pybuda.op.LeakyRelu("lrelu3", mul3, alpha=self.alpha)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", lrelu1, x2)
        mul5 = pybuda.op.Multiply("mul5", lrelu2, x3)
        mul6 = pybuda.op.Multiply("mul6", self.train_param2, lrelu3)

        # Layer 5
        lrelu4 = pybuda.op.LeakyRelu("lrelu4", mul4, alpha=self.alpha)
        lrelu5 = pybuda.op.LeakyRelu("lrelu5", mul5, alpha=self.alpha)
        lrelu6 = pybuda.op.LeakyRelu("lrelu6", mul6, alpha=self.alpha)

        # Layer 6
        mul7 = pybuda.op.Multiply("mul7", lrelu4, mul2)
        mul8 = pybuda.op.Multiply("mul8", lrelu5, mul3)
        mul9 = pybuda.op.Multiply("mul9", lrelu6, mul1)
        mul10 = pybuda.op.Multiply("mul10", lrelu4, lrelu5)

        # Layer 7
        lrelu7 = pybuda.op.LeakyRelu("lrelu7", mul10, alpha=self.alpha)
        lrelu8 = pybuda.op.LeakyRelu("lrelu8", mul8, alpha=self.alpha)
        lrelu9 = pybuda.op.LeakyRelu("lrelu9", mul9, alpha=self.alpha)

        # Layer 8
        mul11 = pybuda.op.Multiply("mul11", mul7, lrelu7)
        mul12 = pybuda.op.Multiply("mul12", lrelu8, mul6)
        mul13 = pybuda.op.Multiply("mul13", mul5, lrelu9)

        # Layer 9
        lrelu10 = pybuda.op.LeakyRelu("lrelu10", mul11, alpha=self.alpha)
        lrelu11 = pybuda.op.LeakyRelu("lrelu11", mul12, alpha=self.alpha)
        lrelu12 = pybuda.op.LeakyRelu("lrelu12", mul13, alpha=self.alpha)

        # Layer 10
        mul14 = pybuda.op.Multiply("mul14", lrelu10, mul8)
        mul15 = pybuda.op.Multiply("mul15", lrelu11, mul9)
        mul16 = pybuda.op.Multiply("mul16", lrelu12, lrelu6)

        # Layer 11
        mul17 = pybuda.op.Multiply("mul17", mul14, lrelu8)
        mul18 = pybuda.op.Multiply("mul18", mul15, mul16)

        # Layer 12
        lrelu13 = pybuda.op.LeakyRelu("lrelu13", mul18, alpha=self.alpha)

        return mul17, lrelu13
