# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


from pickletools import pyunicode
import random
import torch

import pybuda

from pybuda import PyBudaModule, Tensor



class BudaReduceTest(PyBudaModule):
    """
        Buda Test 4

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
        super().__init__("Buda Test 4")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 4"
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(1, 4):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):
        pass

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, x2)
        mul2 = pybuda.op.Multiply("mul2", self.train_param1, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", self.train_param2, self.train_param3)
        mul4 = pybuda.op.Multiply("mul4", x2, x3)

        # Layer 3
        operands = [x1, self.train_param1, mul1, mul2, x2, self.train_param2, mul3, mul4, x3, self.train_param3]
        reds = []
        for i in range(len(operands)):
            reds.append(self.operator(self.opname + str(i + 1), operands[i], self.dim, self.keepdim))
        self.shape = reds[0].shape

        # Layer 4
        mul5 = pybuda.op.Multiply("mul5", reds[0], reds[1])
        mul6 = pybuda.op.Multiply("mul6", reds[1], reds[2])
        mul7 = pybuda.op.Multiply("mul7", reds[4], reds[5])
        mul8 = pybuda.op.Multiply("mul8", reds[7], reds[8])

        # Layer 5
        hvs1 = pybuda.op.Heaviside("hvs1", mul6, reds[3])
        hvs2 = pybuda.op.Heaviside("hvs2", mul7, reds[6])
        hvs3 = pybuda.op.Heaviside("hvs3", mul8, reds[9])

        # Layer 6
        max1 = pybuda.op.Max("max1", mul5, hvs1)
        max2 = pybuda.op.Multiply("max2", reds[4], hvs2)
        max3 = pybuda.op.Multiply("max3", reds[7], hvs3)

        # Layer 7
        add1 = pybuda.op.Add("add1", reds[1], max1)
        add2 = pybuda.op.Add("add2", reds[3], max2)
        add3 = pybuda.op.Add("add3", reds[6], max3)

        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 8
            lenop = len(operands)
            operands = [add1, reds[3], add2, reds[6], add3, reds[9]]
            preds= []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), operands[i], self.dim, self.keepdim))
            # Layer 9
            mul9 = pybuda.op.Multiply("mul9", preds[0], preds[1])
            mul10 = pybuda.op.Multiply("mul10", preds[2], preds[3])
            mul11 = pybuda.op.Multiply("mul11", preds[4], preds[5])
        else:
            # Layer 9
            mul9 = pybuda.op.Multiply("mul9", add1, reds[3])
            mul10 = pybuda.op.Multiply("mul10", add2, reds[6])
            mul11 = pybuda.op.Multiply("mul11", add3, reds[9])
        
        # Layer 10
        mul12 = pybuda.op.Multiply("mul12", mul9, mul10)

        # Layer 11
        add4 = pybuda.op.Add("add4", mul12, mul11)

        return add4

    def values(self):
        return [item.value() for item in self.inputs] 