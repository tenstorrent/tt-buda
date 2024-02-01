# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 10
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
        Buda Test 10

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
        super().__init__("Buda Test 10")

        self.testname = "Comparison Operator, Test 10"
        self.shape = shape
        self.opname = opname
        self.operator = operator
        self.mask = mask
        self.rng_min = rng_min
        self.rng_max = rng_max
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for _ in range(4):
            input_ = torch.rand(*self.shape) * (self.rng_max - self.rng_min) + self.rng_min
            if self.mask:
                input_ *= (1.0 * torch.randint(0, 2, self.shape))
            self.inputs.append(Tensor.create_from_torch(input_))
        for i in range(1, 5):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3, x4):

        # Layer 2
        comp1 = self.operator(self.opname + "1", x1, self.train_param1)
        comp2 = self.operator(self.opname + "2", self.train_param1, x2)
        comp3 = self.operator(self.opname + "3", x2, self.train_param2)
        comp4 = self.operator(self.opname + "4", x3, self.train_param3)
        comp5 = self.operator(self.opname + "5", self.train_param3, x4)
        comp6 = self.operator(self.opname + "6", x4, self.train_param4)

        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", comp1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", comp2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", comp3, x3)
        mul4 = pybuda.op.Multiply("mul4", comp4, self.train_param3)
        mul5 = pybuda.op.Multiply("mul5", comp5, self.train_param4)
        mul6 = pybuda.op.Multiply("mul6", x4, self.train_param4)

        # Layer 4
        comp7 = self.operator(self.opname + "7", mul1, mul2)
        comp8 = self.operator(self.opname + "8", mul2, mul4)
        comp9 = self.operator(self.opname + "9", mul3, mul6)
        comp10 = self.operator(self.opname + "10", mul5, mul6)

        # Layer 5
        mul7 = pybuda.op.Multiply("mul7", comp7, mul2)
        mul8 = pybuda.op.Multiply("mul8", comp8, mul3)
        mul9 = pybuda.op.Multiply("mul9", comp9, mul4)
        mul10 = pybuda.op.Multiply("mul10", comp10, mul6)

        # Layer 6
        mul11 = pybuda.op.Multiply("mul11", mul7, mul3)
        mul12 = pybuda.op.Multiply("mul12", mul8, mul5)
        mul13 = pybuda.op.Multiply("mul13", mul9, mul10)

        # Layer 7
        comp11 = self.operator(self.opname + "11", mul1, mul7)
        comp12 = self.operator(self.opname + "12", mul11, mul8)
        comp13 = self.operator(self.opname + "13", mul12, mul4)
        comp14 = self.operator(self.opname + "14", mul9, mul13)
        comp15 = self.operator(self.opname + "15", mul13, mul10)

        # Layer 8
        mul14 = pybuda.op.Multiply("mul14", comp11, mul11)
        mul15 = pybuda.op.Multiply("mul15", comp12, mul12)
        mul16 = pybuda.op.Multiply("mul16", comp13, mul9)
        mul17 = pybuda.op.Multiply("mul17", comp14, mul13)
        mul18 = pybuda.op.Multiply("mul18", comp15, mul10)

        # Layer 9
        mul19 = pybuda.op.Multiply("mul19", mul14, comp12)
        mul20 = pybuda.op.Multiply("mul20", mul15, comp13)
        mul21 = pybuda.op.Multiply("mul21", mul16, comp14)
        mul22 = pybuda.op.Multiply("mul22", mul17, comp15)
        mul23 = pybuda.op.Multiply("mul23", comp13, mul18)

        # Layer 10
        comp16 = self.operator(self.opname + "16", mul19, mul20)
        comp17 = self.operator(self.opname + "17", mul20, mul21)
        comp18 = self.operator(self.opname + "18", mul22, mul23)

        # Layer 11
        mul24 = pybuda.op.Multiply("mul24", comp16, mul20)
        mul25 = pybuda.op.Multiply("mul25", comp17, mul21)
        mul26 = pybuda.op.Multiply("mul26", comp18, mul23)

        # Layer 12
        mul27 = pybuda.op.Multiply("mul27", mul24, mul25)
        mul28 = pybuda.op.Multiply("mul28", mul21, mul26)

        return mul27, mul28

    def values(self):
        return [item.value() for item in self.inputs]   