# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
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
        Buda Test 5

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
        super().__init__("Buda Test 5")

        self.testname = "Comparison Operator, Test 5"
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
        comp1 = self.operator(self.opname + "1", x1, self.train_param1)
        comp2 = self.operator(self.opname + "2", self.train_param2, x2)
        comp3 = self.operator(self.opname + "3", self.train_param1, self.train_param2)

        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", comp1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", comp2, x2)
        mul3 = pybuda.op.Multiply("mul3", comp3, self.train_param2)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", x1, mul2)
        mul5 = pybuda.op.Multiply("mul5", x2, mul3)

        # Layer 5
        mul6 = pybuda.op.Multiply("mul6", mul1, mul2)
        mul7 = pybuda.op.Multiply("mul7", mul4, mul3)

        # Layer 6
        comp4 = self.operator(self.opname + "4", mul6, mul4)
        comp5 = self.operator(self.opname + "5", mul7, mul5)

        # Layer 7
        mul8 = pybuda.op.Multiply("mul8", mul1, comp4)
        mul9 = pybuda.op.Multiply("mul9", mul7, comp5)
        comp6 = self.operator(self.opname + "6", mul4, mul2)
        comp7 = self.operator(self.opname + "7", mul5, mul3)

        # Layer 8
        mul10 = pybuda.op.Multiply("mul10", comp6, mul7)
        mul11 = pybuda.op.Multiply("mul11", comp7, mul5)

        # Layer 9
        mul12 = pybuda.op.Multiply("mul12", mul8, mul10)
        mul13 = pybuda.op.Multiply("mul13", mul10, mul9)
        mul14 = pybuda.op.Multiply("mul14", mul9, mul11)

        # Layer 10
        comp8 = self.operator(self.opname + "8", mul12, mul13)
        comp9 = self.operator(self.opname + "9", mul13, mul11)

        # Layer 11
        mul15 = pybuda.op.Multiply("mul15", comp8, mul13)
        mul16 = pybuda.op.Multiply("mul16", comp9, mul14)

        return mul15, mul16

    def values(self):
        return [item.value() for item in self.inputs]   