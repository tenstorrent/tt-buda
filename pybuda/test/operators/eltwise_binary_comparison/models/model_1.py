# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
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
        Buda Test 1

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
        super().__init__("Buda Test 1")

        self.testname = "Comparison Operator, Test 1"
        self.shape = shape
        self.opname = opname
        self.operator = operator
        self.mask = mask
        self.rng_min = rng_min
        self.rng_max = rng_max
        
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        input_ = torch.rand(*self.shape) * (self.rng_max - self.rng_min) + self.rng_min
        if self.mask:
            input_ *= (1.0 * torch.randint(0, 2, self.shape))
        self.inputs = [Tensor.create_from_torch(input_)]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        comp = self.operator(self.opname, x, self.train_param)
        # Layer 3
        mul = pybuda.op.Multiply("mul", comp, self.train_param)

        return mul

    def values(self):
        return [item.value() for item in self.inputs]   