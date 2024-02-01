# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 10
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 10

        In this test we have 22 operations, and 3 input tensors and 9 trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Test 10")
        self.testname = "Operator Matmul Test 10"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = pybuda.Parameter(*self.shape, requires_grad=True)
        
        self.train_param7 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param8 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param9 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(9):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):
        
        # Layer 2
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, -1, -2)
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
        tr2 = pybuda.op.Transpose("tr2", self.train_param2, -1, -2)
        mm2 = pybuda.op.Matmul("mm2", x2, tr2)
        tr3 = pybuda.op.Transpose("tr3", self.train_param3, -1, -2)
        mm3 = pybuda.op.Matmul("mm3", x3, tr3)

        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, x2)
        mm5 = pybuda.op.Matmul("mm5", mm2, x3)
        mm6 = pybuda.op.Matmul("mm6", mm3, x2)

        # Layer 4
        tr4 = pybuda.op.Transpose("tr4", self.train_param4, -1, -2)
        mm7 = pybuda.op.Matmul("mm7", mm4, tr4)
        tr5 = pybuda.op.Transpose("tr5", self.train_param5, -1, -2)
        mm8 = pybuda.op.Matmul("mm8", mm5, tr5)
        tr6 = pybuda.op.Transpose("tr6", self.train_param6, -1, -2)
        mm9 = pybuda.op.Matmul("mm9", mm6, tr6)

        # Layer 
        mm10 = pybuda.op.Matmul("mm10", mm2, self.train_param4)

        # Layer 6
        mm11 = pybuda.op.Matmul("mm11", mm1, self.train_param5)
        mm12 = pybuda.op.Matmul("mm12", mm2, self.train_param6)

        # Layer 7
        tr7 = pybuda.op.Transpose("tr7", mm10, -1, -2)
        mm13 = pybuda.op.Matmul("mm13", mm11, tr7)
        mm14 = pybuda.op.Matmul("mm14", mm7, mm9)
        mm15 = pybuda.op.Matmul("mm15", mm8, mm12)

        # Layer 8
        mm16 = pybuda.op.Matmul("mm16", mm13, self.train_param9)
        mm17 = pybuda.op.Matmul("mm17", mm14, self.train_param7)
        tr8 = pybuda.op.Transpose("tr8", self.train_param8, -1, -2)
        mm18 = pybuda.op.Matmul("mm18", mm15, tr8)

        # Layer 9
        mm19 = pybuda.op.Matmul("mm19", mm16, tr1)
        mm20 = pybuda.op.Matmul("mm20", mm17, tr4)

        # Layer 10
        mm21 = pybuda.op.Matmul("mm21", mm19, mm20)

        # Layer 11
        mm22 = pybuda.op.Matmul("mm22", mm21, mm18)

        return mm22

    def values(self):
        return [item.value() for item in self.inputs]