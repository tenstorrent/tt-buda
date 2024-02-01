# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 9
#   Binary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseBinaryTest(PyBudaModule):
    """
        Buda Test 9

        In this test we have 12 operators with three input operands and three trainable operands.

    Args:
        operator (function): PyBuda binary element-wise operator.
        opname (str): Operation name (e.g. add, mul, sub, ...).
                      This name test use to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 9")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 9"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(3):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        op1 = self.operator(self.opname + "1", x1, self.train_param1)
        op2 = self.operator(self.opname + "2", x2, self.train_param2)
        op3 = self.operator(self.opname + "3", x3, self.train_param3)
        
        # Layer 3
        op4 = self.operator(self.opname + "4", x1, self.train_param2)
        op5 = self.operator(self.opname + "5", x2, self.train_param3)

        # Layer 4
        op6 = self.operator(self.opname + "6", op4, self.train_param2)
        op7 = self.operator(self.opname + "7", op2, x3)
        op8 = self.operator(self.opname + "8", op3, self.train_param3)

        # Layer 5
        op9 = self.operator(self.opname + "9", op6, op7)
        op10 = self.operator(self.opname + "10", op5, op8)

        # Layer 6
        op11 = self.operator(self.opname + "11", op9, op10)

        # Layer 7
        op12 = self.operator(self.opname + "12", op1, op11)

        return op12

    def values(self):
        return [item.value() for item in self.inputs]

    