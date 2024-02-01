# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 6
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 6

        In this test we have 15 unary operations, and 3 input tensors and 6 trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 6")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 6"
        self.shape = shape
        self.kwargs = kwargs

        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]

        for i in range(1, 7):
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
        add1 = pybuda.op.Add("add1", un1, un2)
        mul1 = pybuda.op.Multiply("mul1", un3, un4)
        add2 = pybuda.op.Add("add2", un5, un6)

        # Layer 4
        mul2 = pybuda.op.Multiply("mul2", add1, self.train_param6)
        mul3 = pybuda.op.Multiply("mul3", mul1, self.train_param5)
        mul4 = pybuda.op.Multiply("mul4", add2, self.train_param4)

        # Layer 5
        un7 = self.operator(self.opname + "7", mul2, **self.kwargs)
        un8 = self.operator(self.opname + "8", mul3, **self.kwargs)
        un9 = self.operator(self.opname + "9", mul4, **self.kwargs)

        # Layer 6
        mul5 = pybuda.op.Multiply("mul5", un7, self.train_param4)
        mul6 = pybuda.op.Multiply("mul6", un8, self.train_param5)
        mul7 = pybuda.op.Multiply("mul7", un9, self.train_param6)

        # Layer 7
        un10 = self.operator(self.opname + "10", mul5, **self.kwargs)
        un11 = self.operator(self.opname + "11", mul6, **self.kwargs)
        un12 = self.operator(self.opname + "12", mul7, **self.kwargs)

        # Layer 8
        mul8 = pybuda.op.Multiply("mul8", un10, self.train_param1)
        mul9 = pybuda.op.Multiply("mul9", un11, self.train_param2)
        mul10 = pybuda.op.Multiply("mul10", un12, self.train_param3)

        # Layer 9
        un14 = self.operator(self.opname + "14", mul10, **self.kwargs)
        mul11 = pybuda.op.Multiply("mul11", mul8, mul9)

        # Layer 10
        un13 = self.operator(self.opname + "13", mul11, **self.kwargs)

        # Layer 11
        mul12 = pybuda.op.Multiply("mul12", un13, un14)

        # Layer 12
        un15 = self.operator(self.opname + "15", mul12, **self.kwargs)

        return un15

    def values(self):
        return [item.value() for item in self.inputs]