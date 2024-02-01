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
        add1 = pybuda.op.Add("add1", x1, self.train_param1)
        add2 = pybuda.op.Add("add2", x2, self.train_param1)
        add3 = pybuda.op.Add("add3", x3, self.train_param2)
        mul1 = pybuda.op.Multiply("mul1", x2, self.train_param2)
        mul2 = pybuda.op.Multiply("mul2", x3, self.train_param3)

        # Layer 3
        lrelu1 = pybuda.op.LeakyRelu("lrelu1", add1, alpha=self.alpha)
        lrelu2 = pybuda.op.LeakyRelu("lrelu2", add2, alpha=self.alpha)
        lrelu3 = pybuda.op.LeakyRelu("lrelu3", mul1, alpha=self.alpha)
        lrelu4 = pybuda.op.LeakyRelu("lrelu4", add3, alpha=self.alpha)
        lrelu5 = pybuda.op.LeakyRelu("lrelu5", mul2, alpha=self.alpha)

        # Layer 4
        mul3 = pybuda.op.Multiply("mul3", lrelu1, self.train_param1)
        mul4 = pybuda.op.Multiply("mul4", lrelu2, x2)
        mul5 = pybuda.op.Multiply("mul5", lrelu3, self.train_param2)
        mul6 = pybuda.op.Multiply("mul6", lrelu4, x3)
        add4 = pybuda.op.Add("add4", lrelu5, self.train_param3)

        # Layer 5
        lrelu6 = pybuda.op.LeakyRelu("lrelu6", mul3, alpha=self.alpha)
        lrelu7 = pybuda.op.LeakyRelu("lrelu7", mul4, alpha=self.alpha)
        lrelu8 = pybuda.op.LeakyRelu("lrelu8", mul5, alpha=self.alpha)
        lrelu9 = pybuda.op.LeakyRelu("lrelu9", mul6, alpha=self.alpha)
        lrelu10 = pybuda.op.LeakyRelu("lrelu10", add4, alpha=self.alpha)

        # Layer 6
        mul7 = pybuda.op.Multiply("mul7", lrelu6, add2)
        mul8 = pybuda.op.Multiply("mul8", lrelu8, lrelu4)
        mul9 = pybuda.op.Multiply("mul9", lrelu9, lrelu5)
        mul10 = pybuda.op.Multiply("mul10", lrelu10, self.train_param3)
        add5 = pybuda.op.Add("add5", lrelu7, lrelu3)

        # Layer 7
        lrelu11 = pybuda.op.LeakyRelu("lrelu11", mul7, alpha=self.alpha)
        lrelu12 = pybuda.op.LeakyRelu("lrelu12", add5, alpha=self.alpha)
        lrelu13 = pybuda.op.LeakyRelu("lrelu13", mul8, alpha=self.alpha)
        lrelu14 = pybuda.op.LeakyRelu("lrelu14", mul9, alpha=self.alpha)
        lrelu15 = pybuda.op.LeakyRelu("lrelu15", mul10, alpha=self.alpha)

        # Layer 8
        add6 = pybuda.op.Add("add6", lrelu11, mul3)
        add7 = pybuda.op.Add("add7", lrelu12, mul8)
        mul11 = pybuda.op.Multiply("mul11", lrelu13, mul5)
        mul12 = pybuda.op.Multiply("mul12", lrelu14, add4)
        mul13 = pybuda.op.Multiply("mul13", mul5, lrelu15)

        # Layer 9
        lrelu16 = pybuda.op.LeakyRelu("lrelu16", add6, alpha=self.alpha)
        lrelu17 = pybuda.op.LeakyRelu("lrelu17", add7, alpha=self.alpha)
        lrelu18 = pybuda.op.LeakyRelu("lrelu18", mul11, alpha=self.alpha)
        lrelu19 = pybuda.op.LeakyRelu("lrelu19", mul12, alpha=self.alpha)
        lrelu20 = pybuda.op.LeakyRelu("lrelu20", mul13, alpha=self.alpha)

        # Layer 10
        mul14 = pybuda.op.Multiply("mul14", lrelu16, mul7)
        mul15 = pybuda.op.Multiply("mul15", lrelu17, mul8)
        mul16 = pybuda.op.Multiply("mul16", lrelu18, lrelu19)
        mul17 = pybuda.op.Multiply("mul17", add5, lrelu20)

        # Layer 11
        lrelu21 = pybuda.op.LeakyRelu("lrelu21", mul14, alpha=self.alpha)
        lrelu22 = pybuda.op.LeakyRelu("lrelu22", mul15, alpha=self.alpha)
        lrelu23 = pybuda.op.LeakyRelu("lrelu23", mul16, alpha=self.alpha)
        lrelu24 = pybuda.op.LeakyRelu("lrelu24", mul17, alpha=self.alpha)

        # Layer 12
        add8 = pybuda.op.Add("add8", lrelu21, lrelu23)
        add9 = pybuda.op.Add("add9", lrelu22, lrelu24)

        return add8, add9
