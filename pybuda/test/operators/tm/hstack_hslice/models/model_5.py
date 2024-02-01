# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
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
        Buda Test 5

    """

    def __init__(
        self,
        shape, 
        slice):
        super().__init__("Buda Test 5")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 5"
        self.shape = shape
        self.slice = slice

        if type(self.shape) == tuple:
            self.shape = list(self.shape)
        self.shape[1] *= (self.slice * self.slice)
        # self.shape[1] *= self.slice
        # self.shape[-1] *= self.slice
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        hst1 = pybuda.op.HStack("hst1", x1, self.slice)
            # (W, Z, R, C) --> (W, Z // SLICE, R, C * SLICE)
        hst2 = pybuda.op.HStack("hst2", self.train_param1, self.slice)
            # (W, Z, R, C) --> (W, Z // SLICE, R, C * SLICE)
        hst3 = pybuda.op.HStack("hst3", x2, self.slice)
            # (W, Z, R, C) --> (W, Z // SLICE, R, C * SLICE)
        hst4 = pybuda.op.HStack("hst4", self.train_param2, self.slice)
            # (W, Z, R, C) --> (W, Z // SLICE, C * SLICE)

        # Layer 3
        hst5 = pybuda.op.HStack("hst5", hst1, self.slice)
            # (W, Z // SLICE, R, C * SLICE) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)
        hst6 = pybuda.op.HStack("hst6", hst2, self.slice)
            # (W, Z // SLICE, R, C * SLICE) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)
        hst7 = pybuda.op.HStack("hst7", self.train_param2, self.slice)
            # (W, Z, R, C) --> (W, Z // SLICE, R, C * SLICE)
        add1 = pybuda.op.Add("add1", hst3, hst4)
            # (W, Z // SLICE, R, C * SLICE) + (W, Z // SLICE, R, C * SLICE) --> (W, Z // SLICE, R, C * SLICE)

        # Layer 4
        add2 = pybuda.op.Add("add2", hst5, hst6)
            # (W, Z // SLICE ** 2, R, C * SLICE ** 2) + (W, Z // SLICE ** 2, R, C * SLICE ** 2) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)
        hst8 = pybuda.op.HStack("hst8", hst7, self.slice)
            # (W, Z // SLICE, R, C * SLICE) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)
        hst9 = pybuda.op.HStack("hst9", hst8, self.slice)
            # (W, Z // SLICE ** 2, R, C * SLICE ** 2) --> (W, Z // SLICE ** 3, R, C * SLICE ** 3)
        hst10 = pybuda.op.HStack("hst10", add1, self.slice)
            # (W, Z // SLICE, R, C * SLICE) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)

        # Layer 5
        hst11 = pybuda.op.HStack("hst11", add2, self.slice)
            # (W, Z // SLICE ** 2, R, C * SLICE ** 2) --> (W, Z // SLICE ** 3, R, C * SLICE ** 3)
        add3 = pybuda.op.Add("add3", hst11, hst9)
            # (W, Z // SLICE ** 3, R, C * SLICE ** 3) + (W, Z // SLICE ** 3, R, C * SLICE ** 3) --> (W, Z // SLICE ** 3, R, C * SLICE ** 3)
        add4 = pybuda.op.Add("add4", hst10, hst10)
            # (W, Z // SLICE ** 2, R, C * SLICE ** 2) + (W, Z // SLICE ** 2, R, C * SLICE ** 2) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)

        # Layer 6
        hsl1 = pybuda.op.HSlice("hsl1", add3, self.slice)
            # (W, Z // SLICE ** 3, R, C * SLICE ** 3) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)

        # Layer 7
        mul1 = pybuda.op.Multiply("mul1", hsl1, add4)
            # (W, Z // SLICE ** 2, R, C * SLICE ** 2) * (W, Z // SLICE ** 2, R, C * SLICE ** 2) --> (W, Z // SLICE ** 2, R, C * SLICE ** 2)

        return mul1, hst11, hst9

    def values(self):
        return [item.value() for item in self.inputs]   