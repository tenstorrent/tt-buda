# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 5

        In this test we have 7 operations, and 4 input tensors and 4 trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self):
        super().__init__("Buda Test 5")
        self.testname = "Operator Matmul Test 5"

        # Input shapes
        self.shape_input1 = (1, 18, 32, 56)
        self.shape_input2 = (1, 18, 128, 86)
        self.shape_input3 = (1, 18, 256, 100)

        # Trainable parameters shapes
        self.shape_train1 = (1, 18, 56, 128)
        self.shape_train2 = (1, 18, 86, 256)
        self.shape_train3 = (1, 18, 100, 522)
        self.shape_train4 = (1, 18, 256, 128)

        self.train_param1 = pybuda.Parameter(*self.shape_train1, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape_train2, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape_train3, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape_train4, requires_grad=True)

        self.inputs = [
            Tensor.create_from_torch(torch.rand(*self.shape_input1)),
            Tensor.create_from_torch(torch.rand(*self.shape_input2)),
            Tensor.create_from_torch(torch.rand(*self.shape_input3))
        ]
        
        self.set_parameter("train_param1", torch.rand(*self.shape_train1, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape_train2, requires_grad=True))
        self.set_parameter("train_param3", torch.rand(*self.shape_train3, requires_grad=True))
        self.set_parameter("train_param4", torch.rand(*self.shape_train4, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mm1 = pybuda.op.Matmul("mm1", x1, self.train_param1)
                # (1, 18, 32, 56) x (1, 18, 56, 128) -> (1, 18, 32, 128)
        mm2 = pybuda.op.Matmul("mm2", x2, self.train_param2)
                # (1, 18, 128, 86) x (1, 18, 86, 256) -> (1, 18, 128, 256)
        mm3 = pybuda.op.Matmul("mm3", x3, self.train_param3)
                # (1, 18, 256, 100) x (1, 18, 100, 522) -> (1, 18, 256, 522)

        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, mm2)
                # (1, 18, 32, 128) x (1, 18, 128, 256) -> (1, 18, 32, 256)
        mm5 = pybuda.op.Matmul("mm5", mm2, mm3)
                # (1, 18, 128, 256) x (1, 18, 256, 522) -> (1, 18, 128, 522)

        # Layer 4
        mm6 = pybuda.op.Matmul("mm6", mm4, x3)
                # (1, 18, 32, 256) x (1, 18, 256, 100) -> (1, 18, 32, 100)
        inter1 = pybuda.op.Matmul("inter1", mm4, self.train_param4)
                # (1, 18, 32, 256) x (1, 18, 256, 128) -> (1, 18, 32, 128)
        mm7 = pybuda.op.Matmul("mm7", inter1, mm5)
                # (1, 18, 32, 128) x (1, 18, 128, 522) -> (1, 18, 32, 522)

        # Layer 5
        tr1 = pybuda.op.Transpose("tr1", mm6, 3, 2)
                # (1, 18, 32, 100) -> (1, 18, 100, 32)
        mm8 = pybuda.op.Matmul("mm8", tr1, mm7)
                # (1, 18, 100, 32) x (1, 18, 32, 522) -> (1, 18, 100, 522)
        
        return mm8

    def values(self):
        return [item.value() for item in self.inputs]