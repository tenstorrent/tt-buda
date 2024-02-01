# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
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
        Buda Test 2

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
        super().__init__("Buda Test 2")

        self.testname = "Comparison Operator, Test 2"
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
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        comp1 = self.operator(self.opname + "1", self.train_param1, x2)

        # Layer 3
        mul3 = pybuda.op.Multiply("mul3", mul1, comp1)
        mul4 = pybuda.op.Multiply("mul4", comp1, mul2)

        # Layer 4
        comp2 = self.operator(self.opname + "2", mul1, mul3)
        comp3 = self.operator(self.opname + "3", mul4, mul2)

        # Layer 5
        mul5 = pybuda.op.Multiply("mul5", comp2, comp3)

        return mul5

    def values(self):
        return [item.value() for item in self.inputs]   