# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
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
        Buda Test 5

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Buda Test 5")


        self.testname = "Operator Pad, Test 5"
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
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        pad1 = pybuda.op.Pad("pad1", x1, self.pad)
        pad2 = pybuda.op.Pad("pad2", mul1, self.pad)
        pad3 = pybuda.op.Pad("pad3", self.train_param1, self.pad)
        pad4 = pybuda.op.Pad("pad4", x2, self.pad)
        pad5 = pybuda.op.Pad("pad5", mul2, self.pad)
        pad6 = pybuda.op.Pad("pad6", self.train_param2, self.pad)
        pad7 = pybuda.op.Pad("pad7", x3, self.pad)
        pad8 = pybuda.op.Pad("pad8", mul3, self.pad)
        pad9 = pybuda.op.Pad("pad9", self.train_param3, self.pad)

        # Layer 4
        pad10 = pybuda.op.Pad("pad10", x1, self.pad)
        mul4 = pybuda.op.Multiply("mul4", pad1, pad2)
        mul5 = pybuda.op.Multiply("mul5", pad2, pad3)
        mul6 = pybuda.op.Multiply("mul6", pad4, pad5)
        mul7 = pybuda.op.Multiply("mul7", pad5, pad6)
        mul8 = pybuda.op.Multiply("mul8", pad7, pad8)
        mul9 = pybuda.op.Multiply("mul9", pad8, pad9)

        # Layer 5
        mul10 = pybuda.op.Multiply("mul10", pad10, mul4)
        pad11 = pybuda.op.Pad("pad11", x2, self.pad)
        mul11 = pybuda.op.Multiply("mul11", mul5, pad11)
        pad12 = pybuda.op.Pad("pad12", x3, self.pad)
        mul12 = pybuda.op.Multiply("mul12", mul7, pad12)
        pad13 = pybuda.op.Pad("pad13", self.train_param3, self.pad)
        mul13 = pybuda.op.Multiply("mul13", mul9, pad13)

        # Layer 6
        pad14 = pybuda.op.Pad("pad14", mul10, self.pad)
        pad15 = pybuda.op.Pad("pad15", mul11, self.pad)
        pad16 = pybuda.op.Pad("pad16", mul6, self.pad)
        pad17 = pybuda.op.Pad("pad17", mul12, self.pad)
        pad18 = pybuda.op.Pad("pad18", mul8, self.pad)
        pad19 = pybuda.op.Pad("pad19", mul13, self.pad)

        # Layer 7
        mul14 = pybuda.op.Multiply("mul14", pad14, pad15)
        mul15 = pybuda.op.Multiply("mul15", pad16, pad17)
        mul16 = pybuda.op.Multiply("mul16", pad18, pad19)

        # Layer 8
        pad20 = pybuda.op.Pad("pad20", pad14, self.pad)
        pad21 = pybuda.op.Pad("pad21", mul14, self.pad)
        pad22 = pybuda.op.Pad("pad22", pad16, self.pad)
        pad23 = pybuda.op.Pad("pad23", mul15, self.pad)
        pad24 = pybuda.op.Pad("pad24", pad19, self.pad)
        pad25 = pybuda.op.Pad("pad25", mul16, self.pad)

        # Layer 9
        mul17 = pybuda.op.Multiply("mul17", pad20, pad23)
        mul18 = pybuda.op.Multiply("mul18", pad22, pad25)
        mul19 = pybuda.op.Multiply("mul19", pad21, pad24)

        # Layer 10
        pad26 = pybuda.op.Pad("pad26", mul17, self.pad)
        pad27 = pybuda.op.Pad("pad27", mul18, self.pad)
        pad28 = pybuda.op.Pad("pad28", mul19, self.pad)

        # Layer 11
        add1 = pybuda.op.Add("add1", pad26, pad27)
        add2 = pybuda.op.Add("add2", pad27, pad28)

        # Layer 12
        pad29 = pybuda.op.Pad("pad29", add1, self.pad)
        pad30 = pybuda.op.Pad("pad30", add2, self.pad)

        return pad29, pad30