# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 5

        In this test we have 23 unary operations, and three input tensors and three trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 5")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 5"
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
        un3 = self.operator(self.opname + "3", x2, **self.kwargs)
        un4 = self.operator(self.opname + "4", self.train_param2, **self.kwargs)
        un5 = self.operator(self.opname + "5", x3, **self.kwargs)
        un6 = self.operator(self.opname + "6", self.train_param3, **self.kwargs)

        # Layer 3
        add1 = pybuda.op.Add("add1", un4, un5)
        sub1 = pybuda.op.Subtract("sub1", un1, un3)
        add2 = pybuda.op.Add("add2", un2, un6)
        sub2 = pybuda.op.Subtract("sub2", un1, un4)
        add3 = pybuda.op.Add("add3", un3, un6)
        sub3 = pybuda.op.Subtract("sub3", un2, un5)

        # Layer 4
        un7 = self.operator(self.opname + "7", add1, **self.kwargs)
        un8 = self.operator(self.opname + "8", sub1, **self.kwargs)
        un9 = self.operator(self.opname + "9", add2, **self.kwargs)
        un10 = self.operator(self.opname + "10", sub2, **self.kwargs)
        un11 = self.operator(self.opname + "11", add3, **self.kwargs)
        un12 = self.operator(self.opname + "12", sub3, **self.kwargs)

        # Layer 5
        add4 = pybuda.op.Add("add4", un7, self.train_param1)
        mul1 = pybuda.op.Multiply("mul1", un8, un3)
        mul2 = pybuda.op.Multiply("mul2", un9, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", un10, x3)
        add5 = pybuda.op.Add("add5", un11, un6)
        sub4 = pybuda.op.Subtract("sub4", un12, self.train_param3)

        # Layer 6
        un13 = self.operator(self.opname + "13", add4, **self.kwargs)
        un14 = self.operator(self.opname + "14", mul1, **self.kwargs)
        un15 = self.operator(self.opname + "15", mul2, **self.kwargs)
        un16 = self.operator(self.opname + "16", mul3, **self.kwargs)
        un17 = self.operator(self.opname + "17", add5, **self.kwargs)
        un18 = self.operator(self.opname + "18", sub4, **self.kwargs)

        # Layer 7
        add6 = pybuda.op.Add("add6", un13, un14)
        add7 = pybuda.op.Add("add7", un14, un9)
        mul4 = pybuda.op.Multiply("mul4", un15, un10)
        add8 = pybuda.op.Add("add8", un16, un17)
        sub5 = pybuda.op.Subtract("sub5", un11, un18)

        # Layer 8
        add9 = pybuda.op.Add("add9", add6, add7)
        mul5 = pybuda.op.Multiply("mul5", un15, mul4)
        mul6 = pybuda.op.Multiply("mul6", add8, sub5)

        # Layer 9
        un19 = self.operator(self.opname + "19", add9, **self.kwargs)
        un20 = self.operator(self.opname + "20", mul6, **self.kwargs)

        # Layer 10
        mul7 = pybuda.op.Multiply("mul7", un19, mul5)
        mul8 = pybuda.op.Multiply("mul8", mul5, un20)

        # Layer 11
        un21 = self.operator(self.opname + "21", mul7, **self.kwargs)
        un22 = self.operator(self.opname + "22", mul8, **self.kwargs)

        # Layer 12
        add10 = pybuda.op.Add("add10", un21, un22)

        # Layer 13
        un23 = self.operator(self.opname + "23", add10, **self.kwargs)

        return un23

    def values(self):
        return [item.value() for item in self.inputs]