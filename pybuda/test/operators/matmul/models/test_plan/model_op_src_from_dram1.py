# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule


class BudaMatmulTest(PyBudaModule):
    """
        Operator Matmul test - from dram que - parameter

        According to Test plan with this model we are testing:
            1. Op type: Matmul
            2. Operand source: From host + From DRAM queue; we are using pybuda.Parameter as input here
            3. Operand shapes: All cases in combination with this operand source
            4. Operand / output size of dimensions: All cases in combination with this operand source
            5. /
            6. /
        
    """

    def __init__(self, shape):
        super().__init__("Operator Matmul test _ from dram que _ parameter")
        self.testname = "Operator Matmul test _ from dram que _ parameter"

        self.train_param1 = pybuda.Parameter(*shape, requires_grad=True)
        self.set_parameter("train_param1", torch.rand(shape, requires_grad=True))

    def forward(self, x1):

        mm1 = pybuda.op.Matmul("mm1", x1, self.train_param1)   

        return mm1

    def values(self):
        return [item.value() for item in self.inputs]

 