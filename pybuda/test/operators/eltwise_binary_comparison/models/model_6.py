# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 6
#   Cimparison operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaComparisonTest(PyBudaModule):
    """
        Buda Test 6

    """

    def __init__(
        self,
        shape,
        opname,
        operator,
        mask,
        rng_min,
        rng_max
    ):
        super().__init__("Buda Test 6")

        self.testname = "Comparison Operator, Test 6"
        self.shape = shape
        self.opname = opname
        self.operator = operator
        self.mask = mask
        self.rng_min = rng_min
        self.rng_max = rng_max
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for _ in range(3):
            input_ = torch.rand(*self.shape) * (self.rng_max - self.rng_min) + self.rng_min
            if self.mask:
                input_ *= (1.0 * torch.randint(0, 2, self.shape))
            self.inputs.append(Tensor.create_from_torch(input_))
        for i in range(1, 4):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x1, x2)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)
        mul4 = pybuda.op.Multiply("mul4", x2, x3)
        mul5 = pybuda.op.Multiply("mul5", self.train_param2, self.train_param3)

        # Layer 3
        comp1 = self.operator(self.opname + "1", mul1, self.train_param1)
        comp2 = self.operator(self.opname + "2", mul2, x2)
        comp3 = self.operator(self.opname + "3", mul3, x3)
        comp4 = self.operator(self.opname + "4", mul4, mul5)
        comp5 = self.operator(self.opname + "5", mul5, self.train_param3)

        # Layer 4
        mul6 = pybuda.op.Multiply("mul6", comp1, mul2)
        mul7 = pybuda.op.Multiply("mul7", comp2, comp3)
        mul8 = pybuda.op.Multiply("mul8", comp3, mul4)
        mul9 = pybuda.op.Multiply("mul9", comp4, mul5)
        mul10 = pybuda.op.Multiply("mul10", comp4, comp5)

        # Layer 5
        mul11 = pybuda.op.Multiply("mul11", mul6, mul8)
        mul12 = pybuda.op.Multiply("mul12", mul7, mul9)
        mul13 = pybuda.op.Multiply("mul13", mul8, mul10)

        return mul11, mul12, mul13

    def values(self):
        return [item.value() for item in self.inputs]   