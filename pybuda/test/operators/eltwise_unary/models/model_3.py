# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 3

        In this test we have 11 unary operations, and 3 input tensors and 3 trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 3")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 3"
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
        un1 = self.operator(self.opname + "1", x1, **self.kwargs)
        un2 = self.operator(self.opname + "2", self.train_param1, **self.kwargs)
        un3 = self.operator(self.opname + "3", self.train_param2, **self.kwargs)
        un4 = self.operator(self.opname + "4", x3, **self.kwargs)

        # Layer 3
        add1 = pybuda.op.Add("add1", un1, un2)
        mul1 = pybuda.op.Multiply("mul1", x2, un3)
        sub1 = pybuda.op.Subtract("sub1", un4, self.train_param3)

        # Layer 4
        un5 = self.operator(self.opname + "5", add1, **self.kwargs)
        un6 = self.operator(self.opname + "6", mul1, **self.kwargs)
        un7 = self.operator(self.opname + "7", sub1, **self.kwargs)

        # Layer 5
        add2 = pybuda.op.Add("add2", self.train_param3, un7)
        mul2 = pybuda.op.Multiply("mul2", x2, un6)

        # Layer 6
        un8 = self.operator(self.opname + "8", mul2, **self.kwargs)
        un9 = self.operator(self.opname + "9", add2, **self.kwargs)
        sub2 = pybuda.op.Subtract("sub2", self.train_param1, un9)

        # Layer 7
        add3 = pybuda.op.Add("add3", un5, un8)
        un10 = self.operator(self.opname + "10", sub2, **self.kwargs)

        # Layer 8
        mul3 = pybuda.op.Multiply("mul3", add3, un10)

        # Layer 9
        un11 = self.operator(self.opname + "11", mul3, **self.kwargs)

        return un11

    def values(self):
        return [item.value() for item in self.inputs]