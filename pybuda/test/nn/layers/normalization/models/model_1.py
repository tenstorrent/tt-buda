# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 1
# Test for Single Layer Layernorm
#

import torch

import pybuda
import pybuda.op
from pybuda.op import nn

from pybuda import PyBudaModule, Tensor



class LayernormTest(PyBudaModule):

    def __init__(
        self,
        input_shape,
        gamma_shape,
        beta_shape,
        dim,
        epsilon
    ):
        super().__init__("Test 1, Layernorm")

        self.testname = "Layernorm Test 1"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = pybuda.Parameter(*self.gamma_shape, requires_grad=True)
        self.beta = pybuda.Parameter(*self.beta_shape, requires_grad=True)
        self.train_param = pybuda.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.input_shape))]
        self.set_parameter("train_param", torch.rand(*self.input_shape, requires_grad=True))
        self.set_parameter("gamma", torch.rand(*self.gamma_shape, requires_grad=True))
        self.set_parameter("beta", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mul = pybuda.op.Multiply("mul", x, self.train_param) 

        # Layer 3
        ln = nn.Layernorm("ln", mul, self.gamma, self.beta, self.dim, self.epsilon)

        return ln
