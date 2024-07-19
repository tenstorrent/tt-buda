# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


#   Model for testing element-wise binary operators
#   when operand sorce from constants input

import pybuda
import torch

from pybuda import PyBudaModule
from test.operators.utils import ShapeUtils


class BudaElementWiseBinaryTest(PyBudaModule):

    def __init__(self, operator, opname, shape):
        super().__init__("Element-wise binary operator " + opname + " test _ op src const eval pass")
        self.testname = "Element-wise binary operator " + opname + " test _ op src const eval pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()
        
        self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)

        self.add_constant("c1")
        self.set_constant("c1", pybuda.Tensor.create_from_torch(my_rand(*self.constant_shape), constant=True))

        self.add_constant("c2")
        self.set_constant("c2", pybuda.Tensor.create_from_torch(my_rand(*self.constant_shape), constant=True))
       
        self.inputs = [
            pybuda.Tensor.create_from_torch(my_rand(*self.shape))
        ]

    def forward(self, x, y):
        v1 = self.operator(self.opname + "0", self.get_constant("c1"), self.get_constant("c2"))
        # v2 and v3 consume inputs
        v2 = pybuda.op.Add("Add1", x, y)
        v3 = pybuda.op.Add("Add2", v1, v2)
        return v3
