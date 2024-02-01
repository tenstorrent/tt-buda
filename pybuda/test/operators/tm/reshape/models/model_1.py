# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Reshape operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch
import numpy as np

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaReshapeTest(PyBudaModule):
    """
        Buda Test 1

    """

    def __init__(
        self,
        old_shape,
        new_shape):
        super().__init__("Buda Test 1")

        assert np.prod(old_shape) == np.prod(new_shape), "Size of a tensor should stay the same"

        self.testname = "Operator reshape Test 1"
        self.old_shape = old_shape
        self.new_shape = new_shape
        
        self.train_param = pybuda.Parameter(*self.old_shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.old_shape))]
        self.set_parameter("train_param", torch.rand(*self.old_shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x, self.train_param)

        # Layer 3
        rsh1 = pybuda.op.Reshape("rsh1", x, self.new_shape)
        rsh2 = pybuda.op.Reshape("rsh2", self.train_param, self.new_shape)

        # Layer 4
        mul2 = pybuda.op.Multiply("mul2", rsh1, rsh2)

        return mul1, mul2

    def values(self):
        return [item.value() for item in self.inputs]   