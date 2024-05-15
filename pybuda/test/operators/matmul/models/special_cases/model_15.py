# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 15
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaMatmulTest(PyBudaModule):
    """
        Buda Test 15 TODO

        According to pytorch documention for Matmul: https://pytorch.org/docs/stable/generated/torch.matmul.html
        with this model we are testing case: 
            - If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned.
              If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
              If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. 
              The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). For example, if input is a (J x 1 x N x N) tensor and other is a (K x N x N) tensor, out will be a (J x K x N x N) tensor. 
              Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs are broadcastable, and not the matrix dimensions. 
              For example, if input is a (J x 1 x N x M) tensor and other is a (K x M x P) tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the matrix dimensions) are different. out will be a (J x K x N x P) tensor.

        According to Test plan with this model we are testing:
            1. Op type: Matmul
            2. Operand source: From host
            3. Operand shapes: Full tensor (expected shape);
            4. Operand / output size of dimensions: Divisible by 32
            5. /
            6. /
        
    """

    def __init__(self):
        super().__init__("Buda Test 15")
        self.testname = "Operator Matmul Test 15"

        # Input shapes
        # TODO Question: what input shape should we use here to cover this case?
        # when using this inputs, instead of triggering case with input1.shape=(64, ) and input2.shape=(64, 32) we trigger case with inputs: input1.shape=(1, ) and input2.shape=(32, 64) - not what was intended
        # it looks like we cannot trigger case where input shape has only matrix dimensions (no microbatch size)
        # this then produce error: pybuda/pybuda/op/eval/pybuda/matmul.py:29: E    RuntimeError: size mismatch, got input (32), mat (32x64), vec (1) - which is expected for given matrix dimensions, but those are not the ones we wanted to trigger
        self.shape_input1 = (64, )
        self.shape_input2 = (3, 1, 64, 32)

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()
        
        self.inputs = [
            Tensor.create_from_torch(my_rand(*self.shape_input1)),
            Tensor.create_from_torch(my_rand(*self.shape_input2)),
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
