# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


from audioop import add
import random
import torch

import pybuda

from pybuda import PyBudaModule, Tensor



class BudaReduceTest(PyBudaModule):
    """
        Buda Test 5

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
        super().__init__("Buda Test 5")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 5"
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
        
        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        operands = [mul1, self.train_param1, mul2, self.train_param2, mul3, self.train_param3]
        reds = []
        for i in range(len(operands)):
            reds.append(self.operator(self.opname + str(i + 1), self.dim, self.keepdim))
        self.shape = reds[0].shape

        # Layer 4
        add1 = pybuda.op.Add("add1", reds[0], reds[1])
        mul4 = pybuda.op.Multiply("mul4", reds[1], reds[2])
        add2 = pybuda.op.Add("add2", reds[2], reds[3])
        mul5 = pybuda.op.Multiply("mul5", reds[3], reds[4])
        add3 = pybuda.op.Add("add3", reds[4], reds[5])

        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 5
            lenop = len(operands)
            operands = [add1, reds[1], mul4, add2, reds[3], mul5, add3, reds[5]]
            preds = []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), self.dim, self.keepdim))
            self.shape = preds[0].shape
            # Layer 6
            add4 = pybuda.op.Add("add4", preds[0], preds[1])
            sub1 = pybuda.op.Subtract("sub1", preds[2], preds[3])
            max1 = pybuda.op.Max("max1", preds[4], preds[5])
            sub2 = pybuda.op.Subtract("sub2", preds[6], preds[7])
        else:
            # Layer 6
            add4 = pybuda.op.Add("add4", add1, reds[1])
            sub1 = pybuda.op.Subtract("sub1", mul4, add2)
            max1 = pybuda.op.Max("max1", reds[3], mul5)
            sub2 = pybuda.op.Subtract("sub2", add3, reds[5])
        
        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 7
            lenop += len(operands)
            operands = [reds[0], add4, sub1, max1, reds[4], sub2]
            preds = []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), self.dim, self.keepdim))
            # Layer 8
            mul6 = pybuda.op.Multiply("mul6", preds[0], preds[1])
            mul7 = pybuda.op.Multiply("mul7", preds[2], preds[3])
            mul8 = pybuda.op.Multiply("mul8", preds[4], preds[5])
        else:
            # Layer 8
            mul6 = pybuda.op.Multiply("mul6", reds[0], add4)
            mul7 = pybuda.op.Multiply("mul7", sub1, max1)
            mul8 = pybuda.op.Multiply("mul8", reds[4], sub2)

        return mul6, mul7, mul8

    def values(self):
        return [item.value() for item in self.inputs] 