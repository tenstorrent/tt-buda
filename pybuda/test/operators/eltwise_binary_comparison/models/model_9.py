# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 9
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
        Buda Test 9

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
        super().__init__("Buda Test 9")

        self.testname = "Comparison Operator, Test 9"
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
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)
        comp1 = self.operator(self.opname + "1", x1, self.train_param1)
        comp2 = self.operator(self.opname + "2", x2, self.train_param2)
        comp3 = self.operator(self.opname + "3", x3, self.train_param3)

        # Layer 3
        mul4 = pybuda.op.Multiply("mul4", comp1, x2)
        mul5 = pybuda.op.Multiply("mul5", comp2, x3)
        mul6 = pybuda.op.Multiply("mul6", self.train_param3, comp3)

        # Layer 4
        mul7 = pybuda.op.Multiply("mul7", mul1, mul2)
        mul8 = pybuda.op.Multiply("mul8", mul4, mul2)
        mul9 = pybuda.op.Multiply("mul9", mul5, mul3)
        mul10 = pybuda.op.Multiply("mul10", mul2, mul6)

        # Layer 5
        comp4 = self.operator(self.opname + "4", mul7, mul4)
        comp5 = self.operator(self.opname + "5", mul2, mul9)
        comp6 = self.operator(self.opname + "6", mul2, mul10)

        # Layer 6
        mul11 = pybuda.op.Multiply("mul11", comp4, mul8)
        mul12 = pybuda.op.Multiply("mul12", comp5, mul5)
        mul13 = pybuda.op.Multiply("mul13", mul9, mul6)
        mul14 = pybuda.op.Multiply("mul14", comp6, mul10)

        # Layer 7
        comp7 = self.operator(self.opname + "7", mul1, mul11)
        comp8 = self.operator(self.opname + "8", mul11, mul8)
        comp9 = self.operator(self.opname + "9", mul12, mul9)
        comp10 = self.operator(self.opname + "10", mul13, mul10)
        comp11 = self.operator(self.opname + "11", mul14, mul6)

        # Layer 8
        mul15 = pybuda.op.Multiply("mul15", comp7, mul12)
        mul16 = pybuda.op.Multiply("mul16", comp8, mul13)
        mul17 = pybuda.op.Multiply("mul17", comp9, mul14)
        mul18 = pybuda.op.Multiply("mul18", comp10, mul13)
        mul19 = pybuda.op.Multiply("mul19", comp11, mul14)

        # Layer 9
        mul20 = pybuda.op.Multiply("mul20", mul15, mul16)
        mul21 = pybuda.op.Multiply("mul21", mul16, comp9)
        mul22 = pybuda.op.Multiply("mul22", mul17, comp10)
        mul23 = pybuda.op.Multiply("mul23", mul18, mul19)

        # Layer 10
        mul24 = pybuda.op.Multiply("mul24", comp8, mul21)
        mul25 = pybuda.op.Multiply("mul25", comp9, mul22)
        mul26 = pybuda.op.Multiply("mul26", comp11, mul23)

        # Layer 11
        comp12 = self.operator(self.opname + "12", mul20, mul24)
        comp13 = self.operator(self.opname + "13", mul21, mul25)
        comp14 = self.operator(self.opname + "14", mul18, mul26)

        # Layer 12
        mul27 = pybuda.op.Multiply("mul27", comp12, mul21)
        mul28 = pybuda.op.Multiply("mul28", comp13, comp14)

        # Layer 13
        mul29 = pybuda.op.Multiply("mul29", mul27, mul25)
        mul30 = pybuda.op.Multiply("mul30", mul28, mul22)

        return mul29, mul30

    def values(self):
        return [item.value() for item in self.inputs]   