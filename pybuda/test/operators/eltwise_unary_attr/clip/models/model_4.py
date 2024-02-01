# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
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
        min_value,
        max_value
    ):
        super().__init__("Buda Test 4")

        self.testname = "Operator Clip, Test 4"
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
        add1 = pybuda.op.Add("add1", x1, self.train_param1)
        add2 = pybuda.op.Add("add2", x1, x2)
        add3 = pybuda.op.Add("add3", x2, self.train_param3)
        add4 = pybuda.op.Add("add4", x3, self.train_param2)

        # Layer 3
        clip1 = pybuda.op.Clip("clip1", add1, min=self.min_value, max=self.max_value)
        clip2 = pybuda.op.Clip("clip2", add2, min=self.min_value, max=self.max_value)
        clip3 = pybuda.op.Clip("clip3", add3, min=self.min_value, max=self.max_value)
        clip4 = pybuda.op.Clip("clip4", add4, min=self.min_value, max=self.max_value)

        # Layer 4
        clip5 = pybuda.op.Clip("clip5", self.train_param1, min=self.min_value, max=self.max_value)
        clip6 = pybuda.op.Clip("clip6", self.train_param2, min=self.min_value, max=self.max_value)
        clip7 = pybuda.op.Clip("clip7", self.train_param3, min=self.min_value, max=self.max_value)

        # Layer 5
        mul1 = pybuda.op.Multiply("mul1", clip1, clip5)
        mul2 = pybuda.op.Multiply("mul2", clip2, clip3)
        mul3 = pybuda.op.Multiply("mul3", clip5, clip4)
        mul4 = pybuda.op.Multiply("mul4", clip6, clip7)

        # Layer 6
        clip8 = pybuda.op.Clip("clip8", mul1, min=self.min_value, max=self.max_value)
        clip9 = pybuda.op.Clip("clip9", mul2, min=self.min_value, max=self.max_value)
        clip10 = pybuda.op.Clip("clip10", mul3, min=self.min_value, max=self.max_value)
        clip11 = pybuda.op.Clip("clip11", mul4, min=self.min_value, max=self.max_value)

        # Layer 7
        add5 = pybuda.op.Add("add5", clip8, clip5)
        add6 = pybuda.op.Add("add6", clip9, clip6)
        add7 = pybuda.op.Add("add7", clip10, clip7)
        add8 = pybuda.op.Add("add8", clip4, clip11)

        # Layer 8
        clip12 = pybuda.op.Clip("clip12", add5, min=self.min_value, max=self.max_value)
        clip13 = pybuda.op.Clip("clip13", add6, min=self.min_value, max=self.max_value)
        clip14 = pybuda.op.Clip("clip14", add7, min=self.min_value, max=self.max_value)
        clip15 = pybuda.op.Clip("clip15", add8, min=self.min_value, max=self.max_value)

        # Layer 9
        mul5 = pybuda.op.Multiply("mul5", clip1, clip12)
        mul6 = pybuda.op.Multiply("mul6", mul2, clip13)
        mul7 = pybuda.op.Multiply("mul7", clip6, clip14)
        mul8 = pybuda.op.Multiply("mul8", clip15, clip7)

        # Layer 10
        clip16 = pybuda.op.Clip("clip16", mul5, min=self.min_value, max=self.max_value)
        clip17 = pybuda.op.Clip("clip17", mul6, min=self.min_value, max=self.max_value)
        clip18 = pybuda.op.Clip("clip18", mul7, min=self.min_value, max=self.max_value)
        clip19 = pybuda.op.Clip("clip19", mul8, min=self.min_value, max=self.max_value)

        # Layer 11
        mul9 = pybuda.op.Multiply("mul9", clip16, clip17)
        mul10 = pybuda.op.Multiply("mul10", clip17, clip18)
        mul11 = pybuda.op.Multiply("mul11", clip18, clip19)

        # Layer 12
        mul12 = pybuda.op.Multiply("mul12", mul9, clip9)
        mul13 = pybuda.op.Multiply("mul13", mul10, mul11)

        return mul12, mul13
