# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 8
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
        super().__init__("Test 8, Layernorm")

        self.testname = "Layernorm Test 8"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = {
            f"gamma{i}": pybuda.Parameter(*self.gamma_shape, requires_grad=True)
            for i in range(1, 25)
        }
        self.beta = {
            f"beta{i}": pybuda.Parameter(*self.beta_shape, requires_grad=True)
            for i in range(1, 25)
        }
        self.train_param1 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = []
        for i in range(1, 5):
            self.set_parameter(f"train_param{i}", torch.rand(*self.input_shape, requires_grad=True))
            self.inputs.append(Tensor.create_from_torch(torch.rand(*self.input_shape)))
        for i in range(1, 25):
            self.set_parameter(f"gamma{i}", torch.rand(*self.gamma_shape, requires_grad=True))
            self.set_parameter(f"beta{i}", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x1, x2, x3, x4):

        # Layer 2
        ln1 = nn.Layernorm(
            "ln1", 
            x1, 
            self.gamma['gamma1'], 
            self.beta['beta1'], 
            self.dim, 
            self.epsilon
        )
        ln2 = nn.Layernorm(
            "ln2", 
            x2, 
            self.gamma['gamma2'], 
            self.beta['beta2'], 
            self.dim, 
            self.epsilon
        )
        ln3 = nn.Layernorm(
            "ln3", 
            x3, 
            self.gamma['gamma3'], 
            self.beta['beta3'], 
            self.dim, 
            self.epsilon
        )
        ln4 = nn.Layernorm(
            "ln4", 
            x4, 
            self.gamma['gamma4'], 
            self.beta['beta4'], 
            self.dim, 
            self.epsilon
        )

        mul1 = pybuda.op.Multiply("mul1", self.train_param1, self.train_param4)
        mul2 = pybuda.op.Multiply("mul2", self.train_param1, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", self.train_param2, self.train_param3)
        mul4 = pybuda.op.Multiply("mul4", self.train_param3, self.train_param4)

        # Layer 3
        mul5 = pybuda.op.Multiply("mul5", ln1, ln2)
        add1 = pybuda.op.Add("add1", mul1, mul2)
        mul6 = pybuda.op.Multiply("mul6", ln2, ln3)
        add2 = pybuda.op.Add("add2", mul2, mul3)
        add3 = pybuda.op.Add("add3", ln3, ln4)
        add4 = pybuda.op.Add("add4", mul3, mul4)

        # Layer 4
        ln5 = nn.Layernorm(
            "ln5", 
            mul5, 
            self.gamma['gamma5'], 
            self.beta['beta5'], 
            self.dim, 
            self.epsilon
        )
        ln6 = nn.Layernorm(
            "ln6", 
            add1, 
            self.gamma['gamma6'], 
            self.beta['beta6'], 
            self.dim, 
            self.epsilon
        )
        ln7 = nn.Layernorm(
            "ln7", 
            mul6, 
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
        ln9 = nn.Layernorm(
            "ln9", 
            add3, 
            self.gamma['gamma9'], 
            self.beta['beta9'], 
            self.dim, 
            self.epsilon
        )
        ln10 = nn.Layernorm(
            "ln10", 
            add4, 
            self.gamma['gamma10'], 
            self.beta['beta10'], 
            self.dim, 
            self.epsilon
        )

        # Layer 5
        ln11 = nn.Layernorm(
            "ln11", 
            add1, 
            self.gamma['gamma11'], 
            self.beta['beta11'], 
            self.dim, 
            self.epsilon
        )
        ln12 = nn.Layernorm(
            "ln12", 
            mul6, 
            self.gamma['gamma12'], 
            self.beta['beta12'], 
            self.dim, 
            self.epsilon
        )
        ln13 = nn.Layernorm(
            "ln13", 
            add2, 
            self.gamma['gamma13'], 
            self.beta['beta13'], 
            self.dim, 
            self.epsilon
        )
        ln14 = nn.Layernorm(
            "ln14", 
            add3, 
            self.gamma['gamma14'], 
            self.beta['beta14'], 
            self.dim, 
            self.epsilon
        )

        add5 = pybuda.op.Add("add5", ln5, ln11)
        add6 = pybuda.op.Add("add6", ln6, ln12)
        add7 = pybuda.op.Add("add7", ln7, ln13)
        mul7 = pybuda.op.Multiply("mul7", ln8, ln14)
        mul8 = pybuda.op.Multiply("mul8", ln9, ln10)

        # Layer 6
        ln15 = nn.Layernorm(
            "ln15", 
            add5, 
            self.gamma['gamma15'], 
            self.beta['beta15'], 
            self.dim, 
            self.epsilon
        )
        ln16 = nn.Layernorm(
            "ln16", 
            add6, 
            self.gamma['gamma16'], 
            self.beta['beta16'], 
            self.dim, 
            self.epsilon
        )
        ln17 = nn.Layernorm(
            "ln17", 
            add7, 
            self.gamma['gamma17'], 
            self.beta['beta17'], 
            self.dim, 
            self.epsilon
        )
        ln18 = nn.Layernorm(
            "ln18", 
            mul7, 
            self.gamma['gamma18'], 
            self.beta['beta18'], 
            self.dim, 
            self.epsilon
        )
        ln19 = nn.Layernorm(
            "ln19", 
            mul8, 
            self.gamma['gamma19'], 
            self.beta['beta19'], 
            self.dim, 
            self.epsilon
        )

        # Layer 7
        mul9 = pybuda.op.Multiply("mul9", ln5, ln15)
        mul10 = pybuda.op.Multiply("mul10", ln6, ln16)
        add8 = pybuda.op.Add("add8", ln7, ln17)
        add9 = pybuda.op.Add("add9", ln8, ln18)
        add10 = pybuda.op.Add("add10", ln10, ln19)

        # Layer 8
        ln20 = nn.Layernorm(
            "ln20", 
            mul9, 
            self.gamma['gamma20'], 
            self.beta['beta20'], 
            self.dim, 
            self.epsilon
        )
        ln21 = nn.Layernorm(
            "ln21", 
            mul10, 
            self.gamma['gamma21'], 
            self.beta['beta21'], 
            self.dim, 
            self.epsilon
        )
        ln22 = nn.Layernorm(
            "ln22", 
            add9, 
            self.gamma['gamma22'], 
            self.beta['beta22'], 
            self.dim, 
            self.epsilon
        )
        ln23 = nn.Layernorm(
            "ln23", 
            add8, 
            self.gamma['gamma23'], 
            self.beta['beta23'], 
            self.dim, 
            self.epsilon
        )
        ln24 = nn.Layernorm(
            "ln24", 
            add10, 
            self.gamma['gamma24'], 
            self.beta['beta24'], 
            self.dim, 
            self.epsilon
        )

        return ln20, ln21, ln22, ln23, ln24
