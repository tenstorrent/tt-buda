# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 2

        In this test we have 6 unary operations, and three input tensors and three trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 2")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 2"
        self.shape = shape
        self.kwargs = kwargs
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]

        for i in range(1, 4):
            self.set_parameter("train_param{}".format(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        add1 = pybuda.op.Add("add1", x2, self.train_param2)
        mul2 = pybuda.op.Multiply("mul2", x3, self.train_param3)

        # Layer 3
        un1 = self.operator(self.opname + "1", mul1, **self.kwargs)
        un2 = self.operator(self.opname + "2", add1, **self.kwargs)
        un3 = self.operator(self.opname + "3", mul2, **self.kwargs)

        # Layer 4
        mul3 = pybuda.op.Multiply("mul3", un1, un2)

        # Layer 5
        un4 = self.operator(self.opname + "4", mul3, **self.kwargs)
        un5 = self.operator(self.opname + "5", un3, **self.kwargs)

        # Layer 6
        add2 = pybuda.op.Add("add2", un4, un5)

        un6 = self.operator(self.opname + "6", add2, **self.kwargs)

        return un6

    def values(self):
        return [item.value() for item in self.inputs]