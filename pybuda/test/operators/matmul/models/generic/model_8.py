# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 8
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 8

        In this test we have 22 operations, and 3 input tensors and 6 trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Test 8")
        self.testname = "Operator Matmul Test 8"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch((torch.rand(*self.shape) - 0.5).detach()) for i in range(3)]
        for i in range(6):
            self.set_parameter("train_param" + str(i + 1), (torch.rand(*self.shape, requires_grad=True) - 0.5).detach())

    def forward(self, x1, x2, x3):

        # activations, x1, x2, x3
        # (..., H, W)

        # train parameters
        # (..., H, W)

        # Layer 2
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
                # (..., H, W) x (..., W, H) -> (..., H, H)

        tr2 = pybuda.op.Transpose("tr2", self.train_param2, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm2 = pybuda.op.Matmul("mm2", x2, tr2)
                # (..., H, W) x (..., W, H) -> (..., H, H)

        tr3 = pybuda.op.Transpose("tr3", self.train_param3, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm3 = pybuda.op.Matmul("mm3", x3, tr3)
                # (..., H, W) x (..., W, H) -> (..., H, H)
        
        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, self.train_param1)
                # (..., H, H) x (..., H, W) -> (..., H, W)
        tr4 = pybuda.op.Transpose("tr4", x2, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm5 = pybuda.op.Matmul("mm5", tr4, x3)
                # (..., W, H) x (..., H, W) -> (..., W, W)
        mm6 = pybuda.op.Matmul("mm6", mm2, mm3)
                # (..., H, H) x (..., H, H) -> (..., H, H)

        # Layer 4
        mm7 = pybuda.op.Matmul("mm7", mm4, tr4)
                # (..., H, W) x (..., W, H) -> (..., H, H)
        mm8 = pybuda.op.Matmul("mm8", mm4, mm5)
                # (..., H, W) x (..., W, W) -> (..., H, W)
        tr5 = pybuda.op.Transpose("tr5", self.train_param4, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm9 = pybuda.op.Matmul("mm9", mm5, tr5)
                # (..., W, W) x (..., W, H) -> (..., W, H)
        mm10 = pybuda.op.Matmul("mm10", mm6, self.train_param3)
                # (..., H, H) x (..., H, W) -> (..., H, W)

        # Layer 5
        mm11 = pybuda.op.Matmul("mm11", mm7, mm8)
                # (..., H, H) x (..., H, W) -> (..., H, W)
        tr6 = pybuda.op.Transpose("tr6", x3, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm12 = pybuda.op.Matmul("mm12", self.train_param1, tr6)
                # (..., H, W) x (..., W, H) -> (..., H, H)
        mm13 = pybuda.op.Matmul("mm13", mm6, mm10)
                # (..., H, H) x (..., H, W) -> (..., H, W)

        # Layer 6
        tr7 = pybuda.op.Transpose("tr7", mm11, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm14 = pybuda.op.Matmul("mm14", x1, tr7)
                # (..., H, W) x (..., W, H) -> (..., H, H)
        mm15 = pybuda.op.Matmul("mm15", mm12, mm13)
                # (..., H, H) x (..., H, W) -> (..., H, W)
        mm16 = pybuda.op.Matmul("mm16", mm9, self.train_param5)
                # (..., W, H) x (..., H, W) -> (..., W, W)

        # Layer 7
        mm17 = pybuda.op.Matmul("mm17", mm14, mm15)
                # (..., H, H) x (..., H, W) -> (..., H, W)
        tr9 = pybuda.op.Transpose("tr9", self.train_param6, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm18 = pybuda.op.Matmul("mm18", mm15, tr9)
                # (..., H, W) x (..., W, H) -> (..., H, H)
        mm19 = pybuda.op.Matmul("mm19", mm10, mm16)
                # (..., H, W) x (..., W, W) -> (..., H, W)

        # Layer 8
        mm20 = pybuda.op.Matmul("mm20", mm18, mm17)
                # (..., H, H) x (..., H, W) -> (..., H, W)
        mm21 = pybuda.op.Matmul("mm21", mm18, mm19)
                # (..., H, H) x (..., H, W) -> (..., H, W)

        # Layer 9
        tr8 = pybuda.op.Transpose("tr8", mm20, -1, -2)
                # (..., H, W) -> (..., W, H)
        mm22 = pybuda.op.Matmul("mm22", tr8, mm21)
                # (..., W, H) x (..., H, W) -> (..., W, W)
        
        return mm22

    def values(self):
        return [item.value() for item in self.inputs]
