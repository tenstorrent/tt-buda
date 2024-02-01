# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 0
#   Grouped reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaReduceTest(PyBudaModule):
    """
        Buda Test 0

    Args:
        operator (function): PyBuda reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape, dim, groups, keep_dims):
        super().__init__("Buda Test 0")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 0"
        self.shape = shape
        self.dim = dim 
        self.groups = groups
        self.keep_dims = keep_dims
        self.train_param = pybuda.Parameter(torch.randn(*self.shape), requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]

    def forward(self, x):
        mul = pybuda.op.Multiply("mul", x, self.train_param)
        red = self.operator(self.opname, mul, self.dim, self.groups, self.keep_dims)

        return red

    def values(self):
        return [item.value() for item in self.inputs]