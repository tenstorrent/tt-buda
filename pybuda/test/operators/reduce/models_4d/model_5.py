# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


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

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 5")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 5"
        self.shape = shape

        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]

        for i in range(1, 4):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)
        tr2 = pybuda.op.Transpose("tr2", x2, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)
        tr3 = pybuda.op.Transpose("tr3", self.train_param3, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)
        tr4 = pybuda.op.Transpose("tr4", x3, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)

        # Layer 3
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)
        mm2 = pybuda.op.Matmul("mm2", tr2, self.train_param2)
                # (W, Z, C, R) x (W, Z, R, C) --> (W, Z, C, C)
        mm3 = pybuda.op.Matmul("mm3", x3, tr3)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)

        # Layer 4
        red1 = self.operator(self.opname + "1", mm1, 3)
                # (W, Z, R, R) --> (W, Z, R, 1)
        red2 = self.operator(self.opname + "2", mm2, 2)
                # (W, Z, C, C) --> (W, Z, 1, C)
        red3 = self.operator(self.opname + "3", mm3, 2)
                # (W, Z, R, R) --> (W, Z, 1, R)

        # Layer 5
        tr5 = pybuda.op.Transpose("tr5", red2, 3, 2)
                # (W, Z, 1, C) --> (W, Z, C, 1)
        mm4 = pybuda.op.Matmul("mm4", red1, red2)
                # (W, Z, R, 1) x (W, Z, 1, C) --> (W, Z, R, C)
        mm5 = pybuda.op.Matmul("mm5", tr5, red3)
                # (W, Z, C, 1) x (W, Z, 1, R) --> (W, Z, C, R)

        # Layer 6
        red4 = self.operator(self.opname + "4", mm4, 3)
                # (W, Z, R, C) --> (W, Z, R, 1)
        red5 = self.operator(self.opname + "5", mm5, 3)
                # (W, Z, C, R) --> (W, Z, C, 1)
        red6 = self.operator(self.opname + "6", self.train_param3, 2)
                # (W, Z, R, C) --> (W, Z, 1, C)
        red7 = self.operator(self.opname + "7", tr4, 2)
                # (W, Z, C, R) --> (W, Z, 1, R)

        # Layer 7
        mm6 = pybuda.op.Matmul("mm6", red4, red7)
                # (W, Z, R, 1) x (W, Z, 1, C) --> (W, Z, R, C)
        mm7 = pybuda.op.Matmul("mm7", red5, red6)
                # (W, Z, C, 1) x (W, Z, 1, R) --> (W, Z, C, R)

        # Layer 8
        red8 = self.operator(self.opname + "8", mm6, 3)
                # (W, Z, R, C) --> (W, Z, R, 1)
        red9 = self.operator(self.opname + "9", mm7, 2)
                # (W, Z, C, R) --> (W, Z, 1, R)

        # Layer 9
        mm8 = pybuda.op.Matmul("mm8", red8, red9)
                # (W, Z, R, 1) x (W, Z, 1, R) --> (W, Z, R, R)

        return mm8, red8, red9

    def values(self):
        return [item.value() for item in self.inputs]   