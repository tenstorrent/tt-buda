# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 9
#   Unary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Buda Test 9

        In this test we have 11 unary operations, and 2 input tensors and 2 trainable variables.

    Args:
        operator (function): PyBuda unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 9")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 9"
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
        mul1 = pybuda.op.Multiply("mul1", un1, un4)
        mul2 = pybuda.op.Multiply("mul2", un2, un3)

        # Layer 4
        un5 = self.operator(self.opname + "5", mul1, **self.kwargs)
        un6 = self.operator(self.opname + "6", mul2, **self.kwargs)

        # Layer 5
        un7 = self.operator(self.opname + "7", un5, **self.kwargs)
        un8 = self.operator(self.opname + "8", un6, **self.kwargs)

        # Layer 6
        mul3 = pybuda.op.Multiply("mul3", un7, un5)
        mul4 = pybuda.op.Multiply("mul4", un8, un8)

        # Layer 7
        mul5 = pybuda.op.Multiply("mul5", mul3, mul1)
        mul6 = pybuda.op.Multiply("mul6", mul4, mul2)

        # Layer 8
        un9 = self.operator(self.opname + "9", mul5, **self.kwargs)
        un10 = self.operator(self.opname + "10", mul6, **self.kwargs)

        # Layer 9
        mul7 = pybuda.op.Multiply("mul7", un9, mul2)
        mul8 = pybuda.op.Multiply("mul8", mul1, un10)

        # Layer 10
        mul9 = pybuda.op.Multiply("mul9", mul7, mul8)

        # Layer 11
        un11 = self.operator(self.opname + "11", mul9, **self.kwargs)

        return un11

    def values(self):
        return [item.value() for item in self.inputs]