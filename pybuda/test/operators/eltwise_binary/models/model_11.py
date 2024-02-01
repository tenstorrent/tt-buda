# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 11
#   Binary element-wise operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaElementWiseBinaryTest(PyBudaModule):
    """
        Buda Test 11

        In this test we have 22 operators with three input operands and 6 trainable operands.

    Args:
        operator (function): PyBuda binary element-wise operator.
        opname (str): Operation name (e.g. add, mul, sub, ...).
                      This name test use to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 11")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 11"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(6):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        op1 = self.operator(self.opname + "1", x1, self.train_param1)
        op2 = self.operator(self.opname + "2", x2, self.train_param2)
        op3 = self.operator(self.opname + "3", x3, self.train_param3)
        
        # Layer 3
        op4 = self.operator(self.opname + "4", op1, self.train_param1)
        op5 = self.operator(self.opname + "5", x2, x3)
        op6 = self.operator(self.opname + "6", op2, op3)

        # Layer 4
        op7 = self.operator(self.opname + "7", op4, x2)
        op8 = self.operator(self.opname + "8", op4, op5)
        op9 = self.operator(self.opname + "9", op5, self.train_param4)
        op10 = self.operator(self.opname + "10", op6, self.train_param3)

        # Layer 5
        op11 = self.operator(self.opname + "11", op7, op8)
        op12 = self.operator(self.opname + "12", self.train_param1, x3)
        op13 = self.operator(self.opname + "13", op6, op10)

        # Layer 6
        op14 = self.operator(self.opname + "14", x1, op11)
        op15 = self.operator(self.opname + "15", op12, op13)
        op16 = self.operator(self.opname + "16", op9, self.train_param5)

        # Layer 7
        op17 = self.operator(self.opname + "17", op14, op15)
        op18 = self.operator(self.opname + "18", op15, self.train_param6)
        op19 = self.operator(self.opname + "19", op16, op10)

        # Layer 8
        op20 = self.operator(self.opname + "20", op17, op18)
        op21 = self.operator(self.opname + "21", op18, op19)

        # Layer 9
        op22 = self.operator(self.opname + "22", op20, op21)
        
        return op22

    def values(self):
        return [item.value() for item in self.inputs]

    