# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


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

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 4")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 4"
        self.shape = shape

        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]

        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        tr1 = pybuda.op.Transpose("tr1", self.train_param1, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)
        tr2 = pybuda.op.Transpose("tr2", x2, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)

        # Layer 3
        mm1 = pybuda.op.Matmul("mm1", x1, tr1)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)
        mm2 = pybuda.op.Matmul("mm2", self.train_param2, tr2)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)

        # Layer 4
        red1 = self.operator(self.opname + "1", mm1, 3)
                # (W, Z, R, R) --> (W, Z, R, 1)
        red2 = self.operator(self.opname + "2", mm2, 2)
                # (W, Z, R, R) --> (W, Z, 1, R)

        # Layer 5
        mm3 = pybuda.op.Matmul("mm3", red1, red2)
                # (W, Z, R, 1) x (W, Z, 1, R) --> (W, Z, R, R)

        # Layer 6
        red3 = self.operator(self.opname + "3", mm3, 2)
                # (W, Z, R, R) --> (W, Z, 1, R)

        # Layer 7
        tr3 = pybuda.op.Transpose("tr3", red3, 3, 2)
                # (W, Z, 1, R) --> (W, Z, R, 1)

        mm4 = pybuda.op.Matmul("mm4", mm3, tr3)
                # (W, Z, R, R) x (W, Z, R, 1) --> (W, Z, R, 1)
        mm5 = pybuda.op.Matmul("mm5", red3, self.train_param2)
                # (W, Z, 1, R) x (W, Z, R, C) --> (W, Z, 1, C)

        # Layer 8
        mm6 = pybuda.op.Matmul("mm6", mm4, mm5)
                # (W, Z, R, 1) x (W, Z, 1, C) --> (W, Z, R, C)

        # Layer 9
        red4 = self.operator(self.opname + "4", mm6, 2)
                # (W, Z, R, C) --> (W, Z, 1, C)

        return red4

    def values(self):
        return [item.value() for item in self.inputs]   