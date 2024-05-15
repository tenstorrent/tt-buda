# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import pybuda

from pybuda import PyBudaModule


class BudaMatmulTest(PyBudaModule):
    """
        Operator Matmul test - from tm edge

        According to Test plan with this model we are testing:
            1. Op type: Matmul
            2. Operand source: From tm edge: tm -> input
            3. Operand shapes: All cases in combination with this operand source
            4. Operand / output size of dimensions: All cases in combination with this operand source
            5. /
            6. /
        
    """

    def __init__(self):
        super().__init__("Operator Matmul test _ from tm edge")
        self.testname = "Operator Matmul test _ from tm edge"

    def forward(self, x1, x2):

        tr1 = pybuda.op.Transpose("tr1", x1, -1, -2)
        tr2 = pybuda.op.Transpose("tr2", x2, -1, -2)
        mm2 = pybuda.op.Matmul("mm2", tr1, tr2)  

        return mm2

    def values(self):
        return [item.value() for item in self.inputs]
