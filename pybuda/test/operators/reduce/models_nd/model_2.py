# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 

import random
import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaReduceTest(PyBudaModule):
    """
        Buda Test 2

    Args:
        operator (function): PyBuda reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(
        self, 
        operator, 
        opname,
        shape,
        dim,
        keepdim):
        super().__init__("Buda Test 2")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 2"
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        
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
        operands = [x1, mul1, self.train_param1, x2, mul2, self.train_param2]
        reds = []
        for i in range(len(operands)):
            reds.append(self.operator(self.opname + str(i + 1), operands[i], self.dim, self.keepdim))
        self.shape = reds[0].shape

        # Layer 4
        add1 = pybuda.op.Add("add1", reds[0], reds[1])
        add2 = pybuda.op.Add("add2", reds[2], reds[3])
        add3 = pybuda.op.Add("add3", reds[4], reds[5])

        # Layer 5
        mul3 = pybuda.op.Multiply("mul3", add1, add2)
        mul4 = pybuda.op.Multiply("mul4", reds[4], add3)

        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 6
            red7 = self.operator(self.opname + "7", mul3)
            red8 = self.operator(self.opname + "8", mul4)

            return red7, red8
        
        return mul3, mul4

    def values(self):
        return [item.value() for item in self.inputs] 