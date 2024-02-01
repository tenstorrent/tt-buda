# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
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
        Buda Test 3

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Buda Test 3")


        self.testname = "Operator Pad, Test 3"
        self.shape = shape
        self.pad = pad
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]
        
        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", self.train_param1, x2)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        pad1 = pybuda.op.Pad("pad1", mul1, self.pad)
        pad2 = pybuda.op.Pad("pad2", mul2, self.pad)
        pad3 = pybuda.op.Pad("pad3", mul3, self.pad)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", self.train_param1, x2)
        add1 = pybuda.op.Add("add1", x2, self.train_param2)

        # Layer 5
        pad4 = pybuda.op.Pad("pad4", mul4, self.pad)
        pad5 = pybuda.op.Pad("pad5", add1, self.pad)

        # Layer 6
        mul5 = pybuda.op.Multiply("mul5", pad1, pad4)
        mul6 = pybuda.op.Multiply("mul6", pad2, pad3)
        add2 = pybuda.op.Add("add2", pad3, pad5)

        # Layer 7
        pad6 = pybuda.op.Pad("pad6", mul5, self.pad)
        pad7 = pybuda.op.Pad("pad7", mul6, self.pad)
        pad8 = pybuda.op.Pad("pad8", add2, self.pad)

        # Layer 8
        add4 = pybuda.op.Add("add4", pad6, pad7)
        add5 = pybuda.op.Add("add5", pad6, pad8)
        add6 = pybuda.op.Add("add6", pad7, pad8)

        # Layer 9
        pad9 = pybuda.op.Pad("pad9", add4, self.pad)
        pad10 = pybuda.op.Pad("pad10", add5, self.pad)
        pad11 = pybuda.op.Pad("pad11", add6, self.pad)

        return pad9, pad10, pad11