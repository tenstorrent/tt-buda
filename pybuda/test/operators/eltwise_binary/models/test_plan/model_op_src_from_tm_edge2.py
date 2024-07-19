# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


#   Model for testing element-wise binary operators
#   and when operand sorce is from tm edge
#   Combination: - tm -> input


import pybuda

from pybuda import PyBudaModule


class BudaElementWiseBinaryTest(PyBudaModule):

    def __init__(self, operator, opname, shape):
        super().__init__("Element-wise binary operator " + opname + " test _ op src from tm edge2")
        self.testname = "Element-wise binary operator " + opname + " test _ op src from tm edge2"
        self.operator = operator
        self.opname = opname
        self.shape = shape

    def forward(self, x, y):
        # 
        xx = pybuda.op.tm.Transpose("Transpose0", x, -1, -2)
        yy = pybuda.op.tm.Transpose("Transpose1", y, -1, -2)
        output = self.operator(self.opname + "2", xx, yy)
        return output
