# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 10
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 10

        In this test we have 11 unary operations, and 2 input tensors and 2 trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 10")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 10"
        self.shape = shape
        self.kwargs = kwargs
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]

        for i in range(1, 3):
            self.set_parameter("train_param{}".format(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, x1)
        mul2 = pybuda.op.Multiply("mul2", self.train_param1, self.train_param1)
        mul3 = pybuda.op.Multiply("mul3", x2, x2)
        mul4 = pybuda.op.Multiply("mul4", self.train_param2, self.train_param2)

        # Layer 3
        un1 = self.operator(self.opname + "1", mul1, **self.kwargs)
        un2 = self.operator(self.opname + "2", mul2, **self.kwargs)
        un3 = self.operator(self.opname + "3", mul3, **self.kwargs)
        un4 = self.operator(self.opname + "4", mul4, **self.kwargs)

        # Layer 4
        mul5 = pybuda.op.Multiply("mul5", un1, un2)
        mul6 = pybuda.op.Multiply("mul6", un1, un2)
        mul7 = pybuda.op.Multiply("mul7", un1, un2)
        mul8 = pybuda.op.Multiply("mul8", un3, un4)
        mul9 = pybuda.op.Multiply("mul9", un3, un4)

        # Layer 5
        un5 = self.operator(self.opname + "5", mul5, **self.kwargs)
        un6 = self.operator(self.opname + "6", mul6, **self.kwargs)
        un7 = self.operator(self.opname + "7", mul7, **self.kwargs)
        un8 = self.operator(self.opname + "8", mul8, **self.kwargs)
        un9 = self.operator(self.opname + "9", mul9, **self.kwargs)

        # Layer 6
        mul10 = pybuda.op.Multiply("mul10", un5, un6)
        mul11 = pybuda.op.Multiply("mul11", un6, un7)
        mul12 = pybuda.op.Multiply("mul12", un7, un8)
        mul13 = pybuda.op.Multiply("mul13", un8, un9)

        # Layer 7
        un10 = self.operator(self.opname + "10", mul10, **self.kwargs)
        un11 = self.operator(self.opname + "11", mul11, **self.kwargs)
        un12 = self.operator(self.opname + "12", mul12, **self.kwargs)
        un13 = self.operator(self.opname + "13", mul13, **self.kwargs)

        # Layer 8
        mul14 = pybuda.op.Multiply("mul14", un10, un6)
        mul15 = pybuda.op.Multiply("mul15", un11, un7)
        mul16 = pybuda.op.Multiply("mul16", un12, un13)

        # Layer 9
        un14 = self.operator(self.opname + "14", mul14, **self.kwargs)
        un15 = self.operator(self.opname + "15", mul15, **self.kwargs)
        un16 = self.operator(self.opname + "16", mul16, **self.kwargs)

        # Layer 10
        mul17 = pybuda.op.Multiply("mul17", un14, un11)
        mul18 = pybuda.op.Multiply("mul18", un15, un12)

        # Layer 11
        un17 = self.operator(self.opname + "17", mul17, **self.kwargs)
        un18 = self.operator(self.opname + "18", mul18, **self.kwargs)

        # Layer 12
        mul19 = pybuda.op.Multiply("mul19", un17, un18)

        # Layer 13
        un19 = self.operator(self.opname + "19", mul19, **self.kwargs)

        return un19, un16

    def values(self):
        return [item.value() for item in self.inputs]