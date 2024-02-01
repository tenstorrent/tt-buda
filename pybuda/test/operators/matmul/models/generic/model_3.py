# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 3

        In this test we have 10 operations, and three input tensors and three trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Test 3")
        self.testname = "Operator Matmul Test 3"
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(3):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, -1, -2)
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
        tr2 = pybuda.op.Transpose("tr2", self.train_param2, -1, -2)
        mm2 = pybuda.op.Matmul("mm2", x2, tr2)
        tr3 = pybuda.op.Transpose("tr3", x3, -1, -2)
        mm3 = pybuda.op.Matmul("mm3", tr3, self.train_param3)

        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, x2)
        mm5 = pybuda.op.Matmul("mm5", self.train_param2, mm3)
        mm6 = pybuda.op.Matmul("mm6", mm3, tr3)
        
        # Layer 4
        mm7 = pybuda.op.Matmul("mm7", mm2, mm5)
        mm8 = pybuda.op.Matmul("mm8", mm6, x3)
        
        # Layer 5
        mm9 = pybuda.op.Matmul("mm9", mm7, mm8)

        # Layer 6
        tr4 = pybuda.op.Transpose("tr4", mm4, -1, -2)
        mm10 = pybuda.op.Matmul("mm10", tr4, mm9)

        return mm10

    def values(self):
        return [item.value() for item in self.inputs]