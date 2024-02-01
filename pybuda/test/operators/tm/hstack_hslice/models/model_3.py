# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
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
        Buda Test 3

    """

    def __init__(
        self,
        shape, 
        slice):
        super().__init__("Buda Test 3")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 3"
        self.shape = shape
        self.slice = slice

        if type(self.shape) == tuple:
            self.shape = list(self.shape)
        self.shape[1] *= self.slice
        self.shape[-1] *= self.slice
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        hst1 = pybuda.op.HStack("hst1", x1, self.slice)
        # +1
        hst2 = pybuda.op.HStack("hst2", self.train_param1, self.slice)
        # +1
        hst3 = pybuda.op.HStack("hst3", x2, self.slice)
        # +1
        hst4 = pybuda.op.HStack("hst4", self.train_param2, self.slice)
        # +1

        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", hst1, hst2) 
        # +1
        mul2 = pybuda.op.Multiply("mul2", hst3, hst4)
        # +1

        # Layer 4
        hsl1 = pybuda.op.HSlice("hsl1", mul1, self.slice)
        # 0
        mul3 = pybuda.op.Multiply("mul3", hst2, mul2)
        # +1

        # Layer 5
        mul4 = pybuda.op.Multiply("mul4", hsl1, x2)
        # 0

        # Layer 6
        hsl2 = pybuda.op.HSlice("hsl2", mul4, self.slice)
        # -1
        hsl3 = pybuda.op.HSlice("hsl3", mul3, self.slice)
        # 0
        hst5 = pybuda.op.HStack("hst5", self.train_param1, self.slice)
        # +1
        hst6 = pybuda.op.HStack("hst6", self.train_param2, self.slice)
        # +1

        # Layer 7
        # hst7 = pybuda.op.HStack("hst7", hst6, self.slice)

        # Layer 8
        add1 = pybuda.op.Add("add1", hsl2,pybuda.op.HSlice("hsl5", hsl3, self.slice))
        # -1
        mul5 = pybuda.op.Multiply("mul5", hst5, hst6)
        # +1

        # Layer 9
        hst8 = pybuda.op.HStack("hst8", add1, self.slice)
        # 0
        hst9 = pybuda.op.HStack("hst9", hst8, self.slice)
        # +1
        # hst10 = pybuda.op.HStack("hsl10", mul5, self.slice)

        # Layer 10
        sub1 = pybuda.op.Subtract("sub1", hst9, mul5)
        # +1

        # Layer 11
        hst10 = pybuda.op.HStack("hst10", sub1, self.slice)

        return hst10

    def values(self):
        return [item.value() for item in self.inputs]   