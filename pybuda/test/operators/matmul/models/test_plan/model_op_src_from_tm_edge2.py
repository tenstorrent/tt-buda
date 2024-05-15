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
        Operator Matmul test - from operator -> tm -> input

        According to Test plan with this model we are testing:
            1. Op type: Matmul
            2. Operand source: From tm edge: Combination: operator -> tm -> input
            3. Operand shapes: All cases in combination with this operand source
            4. Operand / output size of dimensions: All cases in combination with this operand source
            5. /
            6. /
        
    """

    def __init__(self):
        super().__init__("Operator Matmul test - from operator_tm_input")
        self.testname = "Operator Matmul test - from operator_tm_input"


    def forward(self, x1, x2, x3, x4):
        
        add1 = pybuda.op.Add("add1", x1, x2)
        add2 = pybuda.op.Add("add2", x3, x4)
        tr1 = pybuda.op.Transpose("tr1", add1, -1, -2)
        tr2 = pybuda.op.Transpose("tr2", add2, -1, -2)
        mm1 = pybuda.op.Matmul("mm1", tr1, tr2)  

        return mm1

    def values(self):
        return [item.value() for item in self.inputs]
