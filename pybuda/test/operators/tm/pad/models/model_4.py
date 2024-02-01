# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Pad operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, Tensor


class BudaPadTest(PyBudaModule):
    """
        Buda Test 4

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Buda Test 4")


        self.testname = "Operator Pad, Test 4"
        self.shape = shape
        self.pad = pad
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(3)]
        
        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param3", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        pad1 = pybuda.op.Pad("pad1", x1, self.pad)
        pad2 = pybuda.op.Pad("pad2", self.train_param1, self.pad)
        pad3 = pybuda.op.Pad("pad3", x2, self.pad)
        pad4 = pybuda.op.Pad("pad4", self.train_param2, self.pad)
        pad5 = pybuda.op.Pad("pad5", x3, self.pad)
        pad6 = pybuda.op.Pad("pad6", self.train_param3, self.pad)

        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)

        # Layer 4
        pad7 = pybuda.op.Pad("pad7", mul1, self.pad)
        pad8 = pybuda.op.Pad("pad8", mul2, self.pad)
        pad9 = pybuda.op.Pad("pad9", mul3, self.pad)

        # Layer 5
        mul4 = pybuda.op.Multiply("mul4", pad7, pad1)
        mul5 = pybuda.op.Multiply("mul5", pad2, pad8)
        mul6 = pybuda.op.Multiply("mul6", pad8, pad4)
        mul7 = pybuda.op.Multiply("mul7", pad3, pad9)
        mul8 = pybuda.op.Multiply("mul8", pad5, pad6)

        # Layer 6
        pad10 = pybuda.op.Pad("pad10", pad7, self.pad)
        pad11 = pybuda.op.Pad("pad11", mul4, self.pad)
        pad12 = pybuda.op.Pad("pad12", mul5, self.pad)
        pad13 = pybuda.op.Pad("pad13", mul6, self.pad)
        pad14 = pybuda.op.Pad("pad14", mul7, self.pad)
        pad15 = pybuda.op.Pad("pad15", mul8, self.pad)
        pad16 = pybuda.op.Pad("pad16", pad6, self.pad)

        # Layer 7
        mul9 = pybuda.op.Multiply("mul9", pad10, pad12)
        mul10 = pybuda.op.Multiply("mul10", pad11, pad14)
        mul11 = pybuda.op.Multiply("mul11", pad13, pad15)
        mul12 = pybuda.op.Multiply("mul12", pad15, pad16)

        # Layer 8
        pad17 = pybuda.op.Pad("pad17", mul9, self.pad)
        pad18 = pybuda.op.Pad("pad18", mul10, self.pad)
        pad19 = pybuda.op.Pad("pad19", mul11, self.pad)
        pad20 = pybuda.op.Pad("pad20", mul12, self.pad)

        # Layer 9
        mul13 = pybuda.op.Multiply("mul13", pad17, pad18)
        mul14 = pybuda.op.Multiply("mul14", pad18, pad19)
        mul15 = pybuda.op.Multiply("mul15", pad19, pad20)

        return mul13, mul14, mul15