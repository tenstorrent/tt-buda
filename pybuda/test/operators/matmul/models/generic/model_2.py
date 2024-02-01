# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 2

        In this test we have 5 operations, and three input tensors and three trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Test 2")
        self.testname = "Operator Matmul Test 2"
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
                # (W, Z, R, C) --> (W, Z, C, R)
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)
        tr2 = pybuda.op.Transpose("tr2", self.train_param2, -1, -2)
                # (W, Z, R, C) --> (W, Z, C, R)
        mm2 = pybuda.op.Matmul("mm2", x2, tr2)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)
        tr3 = pybuda.op.Transpose("tr3", self.train_param3, -1, -2)
                # (W, Z, R, C) --> (W, Z, C, R)
        mm3 = pybuda.op.Matmul("mm3", x3, tr3)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)

        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, mm2)
                # (W, Z, R, R) x (W, Z, R, R) --> (W, Z, R, R)

        # Layer 4
        mm5 = pybuda.op.Matmul("mm5", mm4, mm3)
                # (W, Z, R, R) x (W, Z, R, R) --> (W, Z, R, R)

        return mm5

    def values(self):
        return [item.value() for item in self.inputs]