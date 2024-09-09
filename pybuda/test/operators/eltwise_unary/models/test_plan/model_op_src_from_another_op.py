# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import pybuda

from pybuda import PyBudaModule


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Element-wise unary operator test - from another op (calc)

        According to Test plan with this model we are testing:
            1. Op type: One of the element-wise unary operator
            2. Operand source: From another op (calc)
            3. Operand shapes: All cases in combination with this operand source
            4. Operand / output size of dimensions: All cases in combination with this operand source
            5. /
            6. /
        
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Element-wise unary operator " + opname + " test _ from another op _ calc")
        self.testname = "Element-wise unary operator " + opname + " test _ from another op _ calc"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x):
        add1 = pybuda.op.Add("add1", x, x)
        un1 = self.operator(self.opname + "1", add1, **self.kwargs)
        return un1

    def values(self):
        return [item.value() for item in self.inputs]
