# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 7
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 7

        In this test we have 25 operations, and 4 input tensors and 4 trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Test 7")
        self.testname = "Operator Matmul Test 7"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch((torch.rand(*self.shape) - 0.5).detach()) for i in range(4)]
        for i in range(4):
            self.set_parameter("train_param" + str(i + 1), (torch.rand(*self.shape, requires_grad=True) - 0.5).detach())

    def forward(self, x1, x2, x3, x4):
        # Layer 2
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
                # (..., R, C) x (..., C, R) --> (..., R, R)

        tr2 = pybuda.op.Transpose("tr2", self.train_param2, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm2 = pybuda.op.Matmul("mm2", x1, tr2)
                # (..., R, C) x (..., C, R) --> (..., R, R)

        tr3 = pybuda.op.Transpose("tr3", x2, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm3 = pybuda.op.Matmul("mm3", tr3, self.train_param2)
                # (..., C, R) x (..., R, C) --> (..., C, C)
        mm4 = pybuda.op.Matmul("mm4", tr3, self.train_param3)
                # (..., C, R) x (..., R, C) --> (..., C, C)

        tr4 = pybuda.op.Transpose("tr4", self.train_param3, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm5 = pybuda.op.Matmul("mm5", x3, tr4)
                # (..., R, C) x (..., C, R) --> (..., R, R)

        tr5 = pybuda.op.Transpose("tr5", self.train_param4, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm6 = pybuda.op.Matmul("mm6", x4, tr5)
                # (..., R, C) x (..., C, R) --> (..., R, R)

        # Layer 3
        mm7 = pybuda.op.Matmul("mm7", mm1, mm2)
                # (..., R, R) x (..., R, R) --> (..., R, R)
        mm8 = pybuda.op.Matmul("mm8", x2, mm3)
                # (..., R, C) x (..., C, C) --> (..., R, C)
        mm9 = pybuda.op.Matmul("mm9", mm3, mm4)
                # (..., C, C) x (..., C, C) --> (..., C, C)
        mm10 = pybuda.op.Matmul("mm10", mm1, mm5)
                # (..., R, R) x (..., R, R) --> (..., R, R)
        mm11 = pybuda.op.Matmul("mm11", tr2, mm6)
                # (..., C, R) x (..., R, R) --> (..., C, R)

        # Layer 4
        mm12 = pybuda.op.Matmul("mm12", mm7, mm8)
                # (..., R, R) x (..., R, C) --> (..., R, C)
        mm13 = pybuda.op.Matmul("mm13", mm8, mm9)
                # (..., R, C) x (..., C, C) --> (..., R, C)
        mm14 = pybuda.op.Matmul("mm14", mm10, mm8)
                # (..., R, R) x (..., R, C) --> (..., R, C)
        mm15 = pybuda.op.Matmul("mm15", mm8, mm11)
                # (..., R, C) x (..., C, R) --> (..., R, R)

        # Layer 5
        tr6 = pybuda.op.Transpose("tr6", mm13, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm16 = pybuda.op.Matmul("mm16", mm12, tr6)
                # (..., R, C) x (..., C, R) --> (..., R, R)
        mm17 = pybuda.op.Matmul("mm17", mm14, tr6)
                # (..., R, C) x (..., C, R) --> (..., R, R)
        mm18 = pybuda.op.Matmul("mm18", mm15, mm14)
                # (..., R, R) x (..., R, C) --> (..., R, C)
        mm19 = pybuda.op.Matmul("mm19", mm15, mm12)
                # (..., R, R) x (..., R, C) --> (..., R, C)

        # Layer 6
        mm20 = pybuda.op.Matmul("mm20", mm16, mm17)
                # (..., R, R) x (..., R, R) --> (..., R, R)
        mm21 = pybuda.op.Matmul("mm21", mm17, mm18)
                # (..., R, R) x (..., R, C) --> (..., R, C)
        mm22 = pybuda.op.Matmul("mm22", mm16, mm19)
                # (..., R, R) x (..., R, C) --> (..., R, C)

        # Layer 7
        mm23 = pybuda.op.Matmul("mm23", mm20, mm21)
                # (..., R, R) x (..., R, C) --> (..., R, C)
        tr7 = pybuda.op.Transpose("tr7", mm22, -1, -2)
                # (..., R, C) --> (..., C, R)
        mm24 = pybuda.op.Matmul("mm24", mm21, tr7)
                # (..., R, C) x (..., C, R) --> (..., R, R)

        # Layer 8
        mm25 = pybuda.op.Matmul("mm25", mm24, mm23)
                # (..., R, R) x (..., R, C) --> (..., R, C)

        return mm25

    def values(self):
        return [item.value() for item in self.inputs]
