# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
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
        Buda Test 4

    """

    def __init__(
        self,
        shape, 
        slice):
        super().__init__("Buda Test 4")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-2] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator VStack, VSLice, Test 4"
        self.shape = shape
        self.slice = slice

        if type(self.shape) == tuple:
            self.shape = list(self.shape)
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
        vsl1 = pybuda.op.VSlice("vsl1", x1, self.slice)
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        vsl2 = pybuda.op.VSlice("vsl2", mul1, self.slice)
        vsl3 = pybuda.op.VSlice("vsl3", mul2, self.slice)
        vsl4 = pybuda.op.VSlice("vsl4", self.train_param2, self.slice)

        # Layer 4
        add1 = pybuda.op.Add("add1", vsl1, vsl2)
        sub1 = pybuda.op.Subtract("sub1", vsl3, vsl4)

        # Layer 5
        vsl5 = pybuda.op.VSlice("vsl5", add1, self.slice)
        vsl6 = pybuda.op.VSlice("vsl6", sub1, self.slice)

        # Layer 6
        sub2 = pybuda.op.Subtract("sub2", self.train_param1, self.train_param2)
        add2 = pybuda.op.Add("add2", vsl5, vsl6)

        # Layer 7
        vsl7 = pybuda.op.VSlice("vsl7", sub2, self.slice)
        hst1 = pybuda.op.VStack("hst1", add2, self.slice)

        # Layer 8
        add3 = pybuda.op.Add("add3", vsl7, hst1)

        # Layer 9
        vsl8 = pybuda.op.VSlice("vsl8", add3, self.slice)

        return vsl8