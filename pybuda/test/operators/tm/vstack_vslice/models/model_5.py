# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   VStack, VSlice operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaVStackVSliceTest(PyBudaModule):
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
        assert shape[-2] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator VStack, VSLice, Test 5"
        self.shape = shape
        self.slice = slice

        if type(self.shape) == tuple:
            self.shape = list(self.shape)
        self.shape[1] *= (self.slice * self.slice)
        self.shape[1] *= self.slice
        self.shape[-2] *= self.slice

        print(f"SHAPE: {self.shape}")
        print(f"SLICE: {self.slice}")
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        vst1 = pybuda.op.VStack("vst1", x1, self.slice)
        vst2 = pybuda.op.VStack("vst2", self.train_param1, self.slice)
        vst3 = pybuda.op.VStack("vst3", x2, self.slice)
        vst4 = pybuda.op.VStack("vst4", self.train_param2, self.slice)

        # Layer 3
        vst5 = pybuda.op.VStack("vst5", vst1, self.slice)
        vst6 = pybuda.op.VStack("vst6", vst2, self.slice)
        vst7 = pybuda.op.VStack("vst7", self.train_param2, self.slice)
        add1 = pybuda.op.Add("add1", vst3, vst4)

        # Layer 4
        add2 = pybuda.op.Add("add2", vst5, vst6)
        vst8 = pybuda.op.VStack("vst8", vst7, self.slice)
        vst9 = pybuda.op.VStack("vst9", vst8, self.slice)
        vst10 = pybuda.op.VStack("vst10", add1, self.slice)

        # Layer 5
        vst11 = pybuda.op.VStack("vst11", add2, self.slice)
        add3 = pybuda.op.Add("add3", vst11, vst9)
        add4 = pybuda.op.Add("add4", vst10, vst10)

        # Layer 6
        vsl1 = pybuda.op.VSlice("vsl1", add3, self.slice)

        # Layer 7
        mul1 = pybuda.op.Multiply("mul1", vsl1, add4)

        return mul1, vst11, vst9