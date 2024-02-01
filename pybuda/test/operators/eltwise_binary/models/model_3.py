# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Binary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseBinaryTest(PyBudaModule):
    """
        Buda Test 3

        In this test we have 10 operations, and three input tensors and three trainable variables.

    Args:
        operator (function): PyBuda binary element-wise operator.
        opname (str): Operation name (e.g. add, mul, sub, ...).
                      This name test use to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 3")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 3"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(3):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):
        op1 = self.operator(self.opname + "1", x1, self.train_param1)
        op2 = self.operator(self.opname + "2", x2, self.train_param2)
        op3 = self.operator(self.opname + "3", x3, self.train_param3)
        op4 = self.operator(self.opname + "4", op1, x2)
        op5 = self.operator(self.opname + "5", self.train_param2, op3)
        op6 = self.operator(self.opname + "6", op3, self.train_param3)
        op7 = self.operator(self.opname + "7", op2, op5)
        op8 = self.operator(self.opname + "8", op6, x3)
        op9 = self.operator(self.opname + "9", op7, op8)
        op10 = self.operator(self.opname + "10", op4, op9)
        return op10

    def values(self):
        return [item.value() for item in self.inputs]