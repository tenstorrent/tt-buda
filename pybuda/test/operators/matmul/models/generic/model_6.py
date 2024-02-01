# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 6
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 6

        In this test we have 13 operations, and 4 input tensors and 4 trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Test 6")
        self.testname = "Operator Matmul Test 6"
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
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, -1, -2)
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)

        tr2 = pybuda.op.Transpose("tr2", self.train_param2, -1, -2)
        mm2 = pybuda.op.Matmul("mm2", x2, tr2)

        tr3 = pybuda.op.Transpose("tr3", x3, -1, -2)
        mm3 = pybuda.op.Matmul("mm3", tr3, self.train_param3)

        tr4 = pybuda.op.Transpose("tr4", x4, -1, -2)
        mm4 = pybuda.op.Matmul("mm4", tr4, self.train_param4)

        # Layer 3
        mm5 = pybuda.op.Matmul("mm5", mm1, mm2)
        mm6 = pybuda.op.Matmul("mm6", x3, mm3)
        mm7 = pybuda.op.Matmul("mm7", mm3, mm4)
        
        # Layer 4
        mm8 = pybuda.op.Matmul("mm8", mm5, mm6)
        mm9 = pybuda.op.Matmul("mm9", mm3, mm7)
        mm10 = pybuda.op.Matmul("mm10", mm6, mm7)
        
        # Layer 5
        mm11 = pybuda.op.Matmul("mm11", mm8, mm9)
        tr5 = pybuda.op.Transpose("tr5", mm10, -1, -2)
        mm12 = pybuda.op.Matmul("mm12", mm9, tr5)

        # Layer 6
        mm13 = pybuda.op.Matmul("mm13", mm11, mm12)

        return mm13

    def values(self):
        return [item.value() for item in self.inputs]