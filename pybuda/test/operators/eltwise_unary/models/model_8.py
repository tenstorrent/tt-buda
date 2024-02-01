# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 8
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 8

        In this test we have 11 unary operations, and 2 input tensors and 2 trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 8")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 8"
        self.shape = shape
        self.kwargs = kwargs

        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]

        for i in range(1, 3):
            self.set_parameter("train_param{}".format(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        un1 = self.operator(self.opname + "1", x1, **self.kwargs)
        un2 = self.operator(self.opname + "2", self.train_param1, **self.kwargs)
        un3 = self.operator(self.opname + "3", x2, **self.kwargs)
        un4 = self.operator(self.opname + "4", self.train_param2, **self.kwargs)

        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", un1, un1)
        mul2 = pybuda.op.Multiply("mul2", un2, un2)
        mul3 = pybuda.op.Multiply("mul3", un3, un3)
        mul4 = pybuda.op.Multiply("mul4", un4, un4)

        # Layer 4
        un5 = self.operator(self.opname + "5", mul1, **self.kwargs)
        un6 = self.operator(self.opname + "6", mul2, **self.kwargs)
        un7 = self.operator(self.opname + "7", mul3, **self.kwargs)
        un8 = self.operator(self.opname + "8", mul4, **self.kwargs)

        # Layer 5
        mul5 = pybuda.op.Multiply("mul5", un5, un7)
        mul6 = pybuda.op.Multiply("mul6", un6, un8)

        # Layer 6
        un9 = self.operator(self.opname + "9", mul5, **self.kwargs)
        un10 = self.operator(self.opname + "10", mul6, **self.kwargs)

        # Layer 7
        mul7 = pybuda.op.Multiply("mul7", un9, un10)

        # Layer 8
        un11 = self.operator(self.opname + "11", mul7, **self.kwargs)

        return un11

    def values(self):
        return [item.value() for item in self.inputs]