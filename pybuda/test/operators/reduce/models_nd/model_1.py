# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaReduceTest(PyBudaModule):
    """
        Buda Test 1

    Args:
        operator (function): PyBuda reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(
        self, 
        operator, 
        opname,
        shape,
        dim,
        keepdim):
        super().__init__("Buda Test 1")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 1"
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):
        
        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x, self.train_param)

        # Layer 3
        red1 = self.operator(self.opname + "1", x, self.dim, self.keepdim)
        red2 = self.operator(self.opname + "2", mul1, self.dim, self.keepdim)
        red3 = self.operator(self.opname + "3", self.train_param, self.dim, self.keepdim)

        # Layer 4
        mul2 = pybuda.op.Multiply("mul2", red1, red2)

        # Layer 5
        mul3 = pybuda.op.Multiply("mul3", mul2, red3)

        return mul3

    def values(self):
        return [item.value() for item in self.inputs]   