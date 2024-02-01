# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Matmul Test 1

        In this test we have only one operator with two operands.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Buda Matmul Test 1")
        self.testname = "Operator Matmul Test 1"
        self.shape = shape
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):
        tr = pybuda.op.Transpose("tr", self.train_param, -1, -2)
        return pybuda.op.Matmul("mm", x, tr)

    def values(self):
        return [item.value() for item in self.inputs]