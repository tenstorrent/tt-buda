# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


#   Model for testing element-wise binary operators
#   when operand sorce is from dram queue

import pybuda
import torch

from pybuda import PyBudaModule
from test.operators.utils import ShapeUtils


class BudaElementWiseBinaryTest(PyBudaModule):

    def __init__(self, operator, opname, shape):
        super().__init__("Element-wise binary operator " + opname + " test _ op src from dram queue prologued")
        self.testname = "Element-wise binary operator " + opname + " test _ op src from dram queue prologued"
        self.operator = operator
        self.opname = opname
        self.shape = shape

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()

        self.shape_input = ShapeUtils.reduce_microbatch_size(shape)

        self.add_constant("c")
        self.set_constant("c", pybuda.Tensor.create_from_torch(my_rand(*self.shape_input), constant=True))

    def forward(self, x):
        output = self.operator(self.opname + "0", self.get_constant("c"), x)
        return output