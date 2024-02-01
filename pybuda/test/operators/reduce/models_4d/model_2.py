# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Reduce operators defined by PyBuda API
#   These kinds of tests test only single specific operator through different PyBuda architectures
# 


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

    def __init__(self, operator, opname, shape):
        super().__init__("Buda Test 2")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 2"
        self.shape = shape
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        tr = pybuda.op.Transpose("tr", self.train_param, 3, 2)
                # (W, Z, R, C) --> (W, Z, C, R)

        # Layer 3
        mm1 = pybuda.op.Matmul("mm1", x, tr)
                # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)

        # Layer 4
        red1 = self.operator(self.opname + "1", mm1, 2)
                # (W, Z, R, R) --> (W, Z, 1, R)
        red2 = self.operator(self.opname + "2", self.train_param, 3)
                # (W, Z, R, C) --> (W, Z, R, 1)
        # Layer 5
        mm2 = pybuda.op.Matmul("mm2", red1, red2)
                # (W, Z, 1, R) x (W, Z, R, 1) --> (W, Z, 1, 1)

        # Layer 6
        red3 = self.operator(self.opname + "3", mm2, 2)
                # (W, Z, 1, 1) --> (W, Z, 1, 1)

        return red3

    def values(self):
        return [item.value() for item in self.inputs]