# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


#   Model for testing element-wise binary operators
#   when operand sorce is from another operator

import pybuda

from pybuda import PyBudaModule


class BudaElementWiseBinaryTest(PyBudaModule):

    def __init__(self, operator, opname, shape):
        super().__init__("Element-wise binary operator " + opname + " test _ op src from another op")
        self.testname = "Element-wise binary operator " + opname + " test _ op src from another op"
        self.operator = operator
        self.opname = opname
        self.shape = shape

    def forward(self, x, y):
        # we use Add and Subtract operators to create two operands which are inputs for the binary operator
        xx = pybuda.op.Add("Add0", x, y)
        yy = pybuda.op.Subtract("Subtract0", x, y)
        output = self.operator(self.opname + "1", xx, yy)
        return output

    # TODO: check do we need this
    # def values(self):
    #     return [item.value() for item in self.inputs]
