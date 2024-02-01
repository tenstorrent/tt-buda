# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 4
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
        super().__init__("Test 4, Layernorm")

        self.testname = "Layernorm Test 4"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = {
            f"gamma{i}": pybuda.Parameter(*self.gamma_shape, requires_grad=True)
            for i in range(1, 9)
        }
        self.beta = {
            f"beta{i}": pybuda.Parameter(*self.beta_shape, requires_grad=True)
            for i in range(1, 9)
        }
        self.train_param1 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = []
        for i in range(1, 3):
            self.set_parameter(f"train_param{i}", torch.rand(*self.input_shape, requires_grad=True))
            self.inputs.append(Tensor.create_from_torch(torch.rand(*self.input_shape)))
        for i in range(1, 9):
            self.set_parameter(f"gamma{i}", torch.rand(*self.gamma_shape, requires_grad=True))
            self.set_parameter(f"beta{i}", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, x2)
        mul2 = pybuda.op.Multiply("mul2", x1, self.train_param1)
        mul3 = pybuda.op.Multiply("mul3", self.train_param1, self.train_param2)
        mul4 = pybuda.op.Multiply("mul4", x2, self.train_param2)

        # Layer 3
        ln1 = nn.Layernorm(
            "ln1", 
            mul1, 
            self.gamma['gamma1'], 
            self.beta['beta1'], 
            self.dim, 
            self.epsilon
        )
        ln2 = nn.Layernorm(
            "ln2", 
            mul2, 
            self.gamma['gamma2'], 
            self.beta['beta2'], 
            self.dim, 
            self.epsilon
        )
        ln3 = nn.Layernorm(
            "ln3", 
            mul3, 
            self.gamma['gamma3'], 
            self.beta['beta3'], 
            self.dim, 
            self.epsilon
        )
        ln4 = nn.Layernorm(
            "ln4", 
            mul4, 
            self.gamma['gamma4'], 
            self.beta['beta4'], 
            self.dim, 
            self.epsilon
        )

        # Layer 4
        ln5 = nn.Layernorm(
            "ln5", 
            ln1, 
            self.gamma['gamma5'], 
            self.beta['beta5'], 
            self.dim, 
            self.epsilon
        )
        ln6 = nn.Layernorm(
            "ln6", 
            ln3, 
            self.gamma['gamma6'], 
            self.beta['beta6'], 
            self.dim, 
            self.epsilon
        )

        # Layer 5
        mul5 = pybuda.op.Multiply("mul5", ln5, ln2)
        mul6 = pybuda.op.Multiply("mul6", ln6, ln4)

        # Layer 6
        add1 = pybuda.op.Add("add1", mul5, mul6)
        add2 = pybuda.op.Add("add2", ln1, ln4)

        # Layer 7
        ln7 = nn.Layernorm(
            "ln7", 
            add1, 
            self.gamma['gamma7'],
            self.beta['beta7'],
            self.dim, 
            self.epsilon
        )
        ln8 = nn.Layernorm(
            "ln8", 
            add2, 
            self.gamma['gamma8'],
            self.beta['beta8'],
            self.dim, 
            self.epsilon
        )

        return ln7, ln8
