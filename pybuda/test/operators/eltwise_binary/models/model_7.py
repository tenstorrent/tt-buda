# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 7
#   Binary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseBinaryTest(PyBudaModule):
    """
        Buda Test 7

        In this test we have 25 operators with 4 input operands and 4 trainable operands.

    Args:
        operator (function): PyBuda binary element-wise operator.
        opname (str): Operation name (e.g. add, mul, sub, ...).
                      This name test use to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 7")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 7"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(4)]
        for i in range(4):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3, x4):
        
        # Layer 2
        op1 = self.operator(self.opname + "1", x1, self.train_param1)
        op2 = self.operator(self.opname + "2", x1, self.train_param2)
        op3 = self.operator(self.opname + "3", x2, self.train_param2)
        op4 = self.operator(self.opname + "4", x2, self.train_param3)
        op5 = self.operator(self.opname + "5", x3, self.train_param3)
        op6 = self.operator(self.opname + "6", x4, self.train_param4)

        # Layer 3
        op7 = self.operator(self.opname + "7", op1, op2)
        op8 = self.operator(self.opname + "8", op2, op3)
        op9 = self.operator(self.opname + "9", op3, op4)
        op10 = self.operator(self.opname + "10", op3, op5)
        op11 = self.operator(self.opname + "11", op4, op6)

        # Layer 4
        op12 = self.operator(self.opname + "12", op7, op8)
        op13 = self.operator(self.opname + "13", op7, op9)
        op14 = self.operator(self.opname + "14", op8, op10)
        op15 = self.operator(self.opname + "15", op8, op11)

        # Layer 5
        op16 = self.operator(self.opname + "16", op12, op13)
        op17 = self.operator(self.opname + "17", op13, op14)
        op18 = self.operator(self.opname + "18", op14, op15)
        op19 = self.operator(self.opname + "19", op12, op15)

        # Layer 6
        op20 = self.operator(self.opname + "20", op16, op17)
        op21 = self.operator(self.opname + "21", op17, op18)
        op22 = self.operator(self.opname + "22", op18, op19)

        # Layer 7
        op23 = self.operator(self.opname + "23", op20, op21)
        op24 = self.operator(self.opname + "24", op21, op22)

        # Layer 8
        op25 = self.operator(self.opname + "25", op23, op24)

        return op25

    def values(self):
        return [item.value() for item in self.inputs]

    