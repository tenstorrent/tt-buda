# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 4

        In this test we have 7 operations, and 4 input tensors and 4 trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self):
        super().__init__("Buda Test 4")
        self.testname = "Operator Matmul Test 4"

        # Input shapes
        self.shape_input1 = (1, 1, 64, 64)
        self.shape_input2 = (1, 1, 128, 70)
        self.shape_input3 = (1, 1, 350, 420)
        self.shape_input4 = (1, 1, 540, 768)

        # Trainable parameters shapes
        self.shape_train1 = (1, 1, 64, 128)
        self.shape_train2 = (1, 1, 70, 350)
        self.shape_train3 = (1, 1, 420, 540)
        self.shape_train4 = (1, 1, 768, 320)

        self.train_param1 = pybuda.Parameter(*self.shape_train1, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape_train2, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape_train3, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape_train4, requires_grad=True)

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()
        self.inputs = [
            Tensor.create_from_torch(my_rand(*self.shape_input1)),
            Tensor.create_from_torch(my_rand(*self.shape_input2)),
            Tensor.create_from_torch(my_rand(*self.shape_input3)),
            Tensor.create_from_torch(my_rand(*self.shape_input4))
        ]
        
        self.set_parameter("train_param1", my_rand(*self.shape_train1, requires_grad=True))
        self.set_parameter("train_param2", my_rand(*self.shape_train2, requires_grad=True))
        self.set_parameter("train_param3", my_rand(*self.shape_train3, requires_grad=True))
        self.set_parameter("train_param4", my_rand(*self.shape_train4, requires_grad=True))

    def forward(self, x1, x2, x3, x4):

        # Layer 2
        mm1 = pybuda.op.Matmul("mm1", x1, self.train_param1)   
                # (1, 1, 64, 64) x (1, 1, 64, 128) -> (1, 1, 64, 128)
        mm2 = pybuda.op.Matmul("mm2", x2, self.train_param2)   
                # (1, 1, 128, 70) x (1, 1, 70, 350) -> (1, 1, 128, 350)
        mm3 = pybuda.op.Matmul("mm3", x3, self.train_param3)   
                # (1, 1, 350, 420) x (1, 1, 420, 540) -> (1, 1, 350, 540)
        mm4 = pybuda.op.Matmul("mm4", x4, self.train_param4)   
                # (1, 1, 540, 768) x (1, 1, 768, 320) -> (1, 1, 540, 320)

        # Layer 3
        mm5 = pybuda.op.Matmul("mm5", mm1, mm2)    
                # (1, 1, 64, 128) x (1, 1, 128, 350) -> (1, 1, 64, 350)
        mm6 = pybuda.op.Matmul("mm6", mm3, mm4)
                # (1, 1, 350, 540) x (1, 1, 540, 320) -> (1, 1, 350, 320)

        # Layer 4
        mm7 = pybuda.op.Matmul("mm7", mm5, mm6)    
                # (1, 1, 64, 350) x (1, 1, 350, 320) -> (1, 1, 64, 320)

        return mm7

    def values(self):
        return [item.value() for item in self.inputs]
