# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 12
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 12

        According to pytorch documention for Matmul: https://pytorch.org/docs/stable/generated/torch.matmul.html
        With this model we are testing case: 
            - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        
        According to Test plan with this model we are testing:
            1. Op type: Matmul
            2. Operand source: From host
            3. Operand shapes: Full tensor (expected shape);
            4. Operand / output size of dimensions: Prime numbers
            5. /
            6. /
        
    """

    def __init__(self):
        super().__init__("Buda Test 12")
        self.testname = "Operator Matmul Test 12"

        # Input shapes
        # TODO Question: what input shape should we use here to cover this case?
        # when using this inputs, instead of triggering case with input1.shape=(7, 3) and input2.shape=(3, 7) we trigger case with inputs: input1.shape=(1, 3 ) and input2.shape=(1, 7) - not what was intended
        # it looks like we cannot trigger case where input shape has only matrix dimensions (no microbatch size)
        # this then produce error: pybuda/pybuda/op/eval/pybuda/matmul.py:29: E     RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 1x7) - which is expected for given matrix dimensions, but those are not the ones we wanted to trigger
        self.shape_input1 = (7, 3)
        self.shape_input2 = (3, 7)

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()
        self.inputs = [
            Tensor.create_from_torch(my_rand(*self.shape_input1)),
            Tensor.create_from_torch(my_rand(*self.shape_input2))
        ]
        self.shapes = [
            self.shape_input1,
            self.shape_input2
        ]

    def forward(self, x1, x2):

        mm1 = pybuda.op.Matmul("mm1", x1, x2)   
       
        return mm1

    def values(self):
        return [item.value() for item in self.inputs]
