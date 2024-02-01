# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   HStack, HSlice operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaHStackHSliceTest(PyBudaModule):
    """
        Buda Test 2

    """

    def __init__(
        self,
        shape, 
        slice):
        super().__init__("Buda Test 2")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 2"
        self.shape = shape
        self.slice = slice
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        hsl1 = pybuda.op.HSlice("hsl1", mul1, self.slice)
        hsl2 = pybuda.op.HSlice("hsl2", mul2, self.slice)

        # Layer 4
        mul3 = pybuda.op.Multiply("mul3", hsl1, hsl2)
        mul4 = pybuda.op.Multiply("mul4", self.train_param1, self.train_param2)

        # Layer 5
        hst1 = pybuda.op.HStack("hst1", mul3, self.slice)

        # Layer 6
        add1 = pybuda.op.Add("add1", hst1, mul4)

        # Layer 7
        hst2 = pybuda.op.HStack("hst2", add1, self.slice)

        return hst2

    def values(self):
        return [item.value() for item in self.inputs]   