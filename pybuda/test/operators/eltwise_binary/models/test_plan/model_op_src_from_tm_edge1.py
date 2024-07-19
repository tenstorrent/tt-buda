# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


#   Model for testing element-wise binary operators
#   and when operand sorce is from tm edge
#   Combination: operator -> tm -> input


import pybuda

from pybuda import PyBudaModule


class BudaElementWiseBinaryTest(PyBudaModule):

    def __init__(self, operator, opname, shape):
        super().__init__("Element-wise binary operator " + opname + " test _ op src from tm edge1")
        self.testname = "Element-wise binary operator " + opname + " test _ op src from tm edge1"
        self.operator = operator
        self.opname = opname
        self.shape = shape

    def forward(self, x, y):
        xx = pybuda.op.Add("Add0", x, y)
        yy = pybuda.op.tm.Transpose("Transpose0", xx, -1, -2)
        output = self.operator(self.opname + "1", yy, yy)
        return output
