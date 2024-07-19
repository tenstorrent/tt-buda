# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


#   Model for testing element-wise binary operators
#   when operand sorce is from host

from pybuda import PyBudaModule


class BudaElementWiseBinaryTest(PyBudaModule):

    def __init__(self, operator, opname, shape):
        super().__init__("Element-wise binary operator " + opname + " test _ op src from host")
        self.testname = "Element-wise binary operator " + opname + " test _ op src from host"
        self.operator = operator
        self.opname = opname
        self.shape = shape

    def forward(self, x, y):        
        output = self.operator(self.opname + "0", x, y)
        return output
