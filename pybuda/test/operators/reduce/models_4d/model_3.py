# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


import torch

import pybuda

from pybuda import PyBudaModule, Tensor


class BudaReduceTest(PyBudaModule):
    """
        Buda Test 3

    Args:
        operator (function): PyBuda reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 3")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 3"
        self.shape = shape

        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]

        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        red1 = self.operator(self.opname + "1", x1, 3)
                # (W, Z, R, C) --> (W, Z, R, 1)
        red2 = self.operator(self.opname + "2", self.train_param1, 2)
                # (W, Z, R, C) --> (W, Z, 1, C)
        tr = pybuda.op.Transpose("tr", x2, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)
        red3 = self.operator(self.opname + "3", self.train_param2, 3)
                # (W, Z, R, C) --> (W, Z, R, 1)

        # Layer 3
        mm1 = pybuda.op.Matmul("mm1", red1, red2)
                # (W, Z, R, 1) x (W, Z, 1, C) --> (W, Z, R, C)
        red4 = self.operator(self.opname + "4", tr, 2)
                # (W, Z, C, R) --> (W, Z, 1, R)
        red5 = self.operator(self.opname + "5", red3, 2)
                # (W, Z, R, 1) --> (W, Z, 1, 1)

        # Layer 4
        red6 = self.operator(self.opname + "6", mm1, 3)
                # (W, Z, R, C) --> (W, Z, R, 1)
        mm2 = pybuda.op.Matmul("mm2", red6, red4)
                # (W, Z, R, 1) x (W, Z, 1, R) --> (W, Z, R, R)

        # Layer 5
        red7 = self.operator(self.opname + "7", mm2, 3)
                # (W, Z, R, R) --> (W, Z, R, 1)

        # Layer 6
        mm3 = pybuda.op.Matmul("mm3", red7, red5)
                # (W, Z, R, 1) x (W, Z, 1, 1) --> (W, Z, R, 1)

        # Layer 7
        red8 = self.operator(self.opname + "8", mm3, 2)
                # (W, Z, R, 1) --> (W, Z, 2)

        return red8

    def values(self):
        return [item.value() for item in self.inputs]   