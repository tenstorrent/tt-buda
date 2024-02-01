# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
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
        Buda Test 3

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
        super().__init__("Buda Test 3")

        self.testname = "Comparison Operator, Test 3"
        self.shape = shape
        self.opname = opname
        self.operator = operator
        self.mask = mask
        self.rng_min = rng_min
        self.rng_max = rng_max
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for _ in range(2):
            input_ = torch.rand(*self.shape) * (self.rng_max - self.rng_min) + self.rng_min
            if self.mask:
                input_ *= (1.0 * torch.randint(0, 2, self.shape))
            self.inputs.append(Tensor.create_from_torch(input_))
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        comp1 = self.operator(self.opname + "1", self.train_param1, x2)
        comp2 = self.operator(self.opname + "2", x2, self.train_param2)
        comp3 = self.operator(self.opname + "3", self.train_param1, self.train_param2)

        # Layer 3
        mul2 = pybuda.op.Multiply("mul2", mul1, comp1)
        mul3 = pybuda.op.Multiply("mul3", x2, comp2)
        mul4 = pybuda.op.Multiply("mul4", self.train_param2, comp3)

        # Layer 4
        mul5 = pybuda.op.Multiply("mul5", mul2, comp2)
        mul6 = pybuda.op.Multiply("mul6", comp1, mul3)
        mul7 = pybuda.op.Multiply("mul7", comp2, mul4)
        mul8 = pybuda.op.Multiply("mul8", x2, comp3)

        # Layer 5
        comp4 = self.operator(self.opname + "4", mul5, mul6)
        comp5 = self.operator(self.opname + "5", mul6, mul7)
        comp6 = self.operator(self.opname + "6", mul7, mul8)
        comp7 = self.operator(self.opname + "7", mul4, self.train_param2)

        # Layer 6
        mul9 = pybuda.op.Multiply("mul9", comp4, mul6)
        mul10 = pybuda.op.Multiply("mul10", comp5, mul4)
        mul11 = pybuda.op.Multiply("mul11", comp6, comp3)
        mul12 = pybuda.op.Multiply("mul12", mul8, comp7)

        # Layer 7
        mul13 = pybuda.op.Multiply("mul13", mul9, mul10)
        mul14 = pybuda.op.Multiply("mul14", mul11, mul12)

        return mul13, mul14

    def values(self):
        return [item.value() for item in self.inputs]   