# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 1

        In this test we have only one operator with two operands.
        One operand represents input and the other one is trainable paramater.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 1")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 1"
        self.shape = shape
        self.kwargs = kwargs
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):
        mul = pybuda.op.Multiply("mul", x, self.train_param)
        un = self.operator(self.opname, mul, **self.kwargs)
        return un

    def values(self):
        return [item.value() for item in self.inputs]