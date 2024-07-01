# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Matmul operator defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule

from test.operators.utils.utils import ShapeUtils


class BudaElementWiseUnaryTest(PyBudaModule):
    """
        Element-wise unary operator test _ Const Inputs _ const eval pass"

        According to Test plan with this model we are testing:
            1. Op type: One of the element-wise unary operator
            2. Operand source: Const Inputs: (const eval pass)
            3. Operand shapes: All cases in combination with this operand source
            4. Operand / output size of dimensions: All cases in combination with this operand source
            5. /
            6. /
        
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Element-wise unary operator " + opname + " test_Const Inputs_Const eval pass")
        self.testname = "Element-wise unary operator " + opname + " test_Const Inputs_Const eval pass"
        self.operator = operator
        self.opname = opname
        self.shape = ShapeUtils.reduce_microbatch_size(shape)
        self.kwargs = kwargs

        self.add_constant("c1")
        self.set_constant("c1", pybuda.Tensor.create_from_torch(torch.rand(*self.shape, requires_grad=False), constant=True))

    def forward(self, x):
        un1 = self.operator(self.opname + "1", self.get_constant("c1"), **self.kwargs)
        un2 = self.operator(self.opname + "2", x, **self.kwargs)
        add1 = pybuda.op.Add("add1",un1, un2)
        return add1

    def values(self):
        return [item.value() for item in self.inputs]
