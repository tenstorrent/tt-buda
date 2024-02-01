# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
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
        Buda Test 5

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
        super().__init__("Buda Test 5")

        self.testname = "Operator Clip, Test 5"
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = BudaClipTest.INPUTS_DISTRIBUTION(
                BudaClipTest.INPUTS_RANGE_MIN, 
                BudaClipTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = BudaClipTest.WEIGHTS_DISTRIBUTION(
                BudaClipTest.WEIGHTS_RANGE_MIN, 
                BudaClipTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)



    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        mul4 = pybuda.op.Multiply("mul4", x2, self.train_param1)
        mul5 = pybuda.op.Multiply("mul5", x3, self.train_param2)
        clip1 = pybuda.op.Clip("clip1", mul1, min=self.min_value, max=self.max_value)
        clip2 = pybuda.op.Clip("clip2", mul2, min=self.min_value, max=self.max_value)
        clip3 = pybuda.op.Clip("clip3", mul3, min=self.min_value, max=self.max_value)

        # Layer 4
        clip4 = pybuda.op.Clip("clip4", mul4, min=self.min_value, max=self.max_value)
        clip5 = pybuda.op.Clip("clip5", mul5, min=self.min_value, max=self.max_value)

        # Layer 5
        add1 = pybuda.op.Add("add1", clip1, self.train_param1)
        add2 = pybuda.op.Add("add2", clip4, x2)
        add3 = pybuda.op.Add("add3", clip2, self.train_param2)
        add4 = pybuda.op.Add("add4", clip5, x3)
        add5 = pybuda.op.Add("add5", clip3, self.train_param3)

        # Layer 6
        clip6 = pybuda.op.Clip("clip6", add1, min=self.min_value, max=self.max_value)
        clip7 = pybuda.op.Clip("clip7", add2, min=self.min_value, max=self.max_value)
        clip8 = pybuda.op.Clip("clip8", add3, min=self.min_value, max=self.max_value)
        clip9 = pybuda.op.Clip("clip9", add4, min=self.min_value, max=self.max_value)
        clip10 = pybuda.op.Clip("clip10", add5, min=self.min_value, max=self.max_value)

        # Layer 7
        mul6 = pybuda.op.Multiply("mul6", clip6, clip4)
        mul7 = pybuda.op.Multiply("mul7", mul1, clip7)
        mul8 = pybuda.op.Multiply("mul8", mul2, clip8)
        mul9 = pybuda.op.Multiply("mul9", clip3, clip9)
        mul10 = pybuda.op.Multiply("mul10", add3, clip10)

        # Layer 8
        clip11 = pybuda.op.Clip("clip11", mul6, min=self.min_value, max=self.max_value)
        clip12 = pybuda.op.Clip("clip12", mul7, min=self.min_value, max=self.max_value)
        clip13 = pybuda.op.Clip("clip13", mul8, min=self.min_value, max=self.max_value)
        clip14 = pybuda.op.Clip("clip14", mul9, min=self.min_value, max=self.max_value)
        clip15 = pybuda.op.Clip("clip15", mul10, min=self.min_value, max=self.max_value)

        # Layer 9
        mul11 = pybuda.op.Multiply("mul11", clip11, clip8)
        mul12 = pybuda.op.Multiply("mul12", clip12, clip5)
        mul13 = pybuda.op.Multiply("mul13", clip13, clip7)
        mul14 = pybuda.op.Multiply("mul14", clip14, add5)
        mul15 = pybuda.op.Multiply("mul15", clip13, mul5)

        # Layer 10
        clip16 = pybuda.op.Clip("clip16", mul11, min=self.min_value, max=self.max_value)
        clip17 = pybuda.op.Clip("clip17", mul12, min=self.min_value, max=self.max_value)
        clip18 = pybuda.op.Clip("clip18", mul13, min=self.min_value, max=self.max_value)
        clip19 = pybuda.op.Clip("clip19", mul14, min=self.min_value, max=self.max_value)
        clip20 = pybuda.op.Clip("clip20", mul15, min=self.min_value, max=self.max_value)

        # Layer 11
        mul16 = pybuda.op.Multiply("mul16", clip16, clip12)
        mul17 = pybuda.op.Multiply("mul17", clip17, clip13)
        mul18 = pybuda.op.Multiply("mul18", clip18, clip19)
        mul19 = pybuda.op.Multiply("mul19", clip13, clip20)

        # Layer 12
        clip21 = pybuda.op.Clip("clip21", mul16, min=self.min_value, max=self.max_value)
        clip22 = pybuda.op.Clip("clip22", mul17, min=self.min_value, max=self.max_value)
        clip23 = pybuda.op.Clip("clip23", mul18, min=self.min_value, max=self.max_value)
        clip24 = pybuda.op.Clip("clip24", mul19, min=self.min_value, max=self.max_value)

        # Layer 13
        mul20 = pybuda.op.Multiply("mul20", clip21, mul12)
        mul21 = pybuda.op.Multiply("mul21", clip22, clip18)
        mul22 = pybuda.op.Multiply("mul22", clip23, clip24)

        return mul20, mul21, mul22
