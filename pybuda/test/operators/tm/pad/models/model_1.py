# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
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
        Buda Test 1

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Buda Test 1")


        self.testname = "Operator Pad, Test 1"
        self.shape = shape
        self.pad = pad
        
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mul = pybuda.op.Multiply("mul", x, self.train_param)

        # Layer 3
        pad = pybuda.op.Pad("pad", mul, self.pad)

        return pad