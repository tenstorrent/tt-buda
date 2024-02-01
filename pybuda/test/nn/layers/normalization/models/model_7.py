# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 7
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
        super().__init__("Test 7, Layernorm")

        self.testname = "Layernorm Test 7"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = {
            f"gamma{i}": pybuda.Parameter(*self.gamma_shape, requires_grad=True)
            for i in range(1, 21)
        }
        self.beta = {
            f"beta{i}": pybuda.Parameter(*self.beta_shape, requires_grad=True)
            for i in range(1, 21)
        }
        self.train_param1 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = []
        for i in range(1, 4):
            self.set_parameter(f"train_param{i}", torch.rand(*self.input_shape, requires_grad=True))
            self.inputs.append(Tensor.create_from_torch(torch.rand(*self.input_shape)))
        for i in range(1, 21):
            self.set_parameter(f"gamma{i}", torch.rand(*self.gamma_shape, requires_grad=True))
            self.set_parameter(f"beta{i}", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x1, x2, x3):

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

        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", ln1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", ln2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", ln3, self.train_param3)

        # Layer 4
        ln4 = nn.Layernorm(
            "ln4", 
            mul1, 
            self.gamma['gamma4'], 
            self.beta['beta4'], 
            self.dim, 
            self.epsilon
        )
        ln5 = nn.Layernorm(
            "ln5", 
            mul2, 
            self.gamma['gamma5'], 
            self.beta['beta5'], 
            self.dim, 
            self.epsilon
        )
        ln6 = nn.Layernorm(
            "ln6", 
            mul3, 
            self.gamma['gamma6'], 
            self.beta['beta6'], 
            self.dim, 
            self.epsilon
        )

        # Layer 5
        add1 = pybuda.op.Add("add1", ln4, ln2)
        add2 = pybuda.op.Add("add2", ln5, ln3)
        add3 = pybuda.op.Add("add3", ln6, self.train_param3)

        # Layer 6
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
        ln9 = nn.Layernorm(
            "ln9",
            add3,
            self.gamma['gamma9'],
            self.beta['beta9'],
            self.dim, 
            self.epsilon
        )

        # Layer 7
        mul4 = pybuda.op.Multiply("mul4", ln1, ln7)
        mul5 = pybuda.op.Multiply("mul5", ln2, ln8)
        add4 = pybuda.op.Add("add4", ln3, ln9)

        # Layer 8
        ln10 = nn.Layernorm(
            "ln10", 
            mul4, 
            self.gamma['gamma10'],
            self.beta['beta10'], 
            self.dim, 
            self.epsilon
        )
        ln11 = nn.Layernorm(
            "ln11", 
            mul5, 
            self.gamma['gamma11'], 
            self.beta['beta11'], 
            self.dim, 
            self.epsilon
        )
        ln12 = nn.Layernorm(
            "ln12", 
            add4, 
            self.gamma['gamma12'], 
            self.beta['beta12'], 
            self.dim, 
            self.epsilon
        )

        # Layer 9
        mul6 = pybuda.op.Multiply("mul6", ln10, ln4)
        mul7 = pybuda.op.Multiply("mul7", ln11, ln5)
        mul8 = pybuda.op.Multiply("mul8", ln12, ln6)

        # Layer 10
        mul9 = pybuda.op.Multiply("mul9", mul6, ln5)
        mul10 = pybuda.op.Multiply("mul10", ln2, mul8)
        add5 = pybuda.op.Add("add5", ln4, mul7)

        # Layer 11
        ln13 = nn.Layernorm(
            "ln13", 
            mul9, 
            self.gamma['gamma13'], 
            self.beta['beta13'], 
            self.dim, 
            self.epsilon
        )
        ln14 = nn.Layernorm(
            "ln14", 
            add5, 
            self.gamma['gamma14'], 
            self.beta['beta14'], 
            self.dim, 
            self.epsilon
        )
        ln15 = nn.Layernorm(
            "ln15", 
            mul10, 
            self.gamma['gamma15'], 
            self.beta['beta15'], 
            self.dim, 
            self.epsilon
        )

        # Layer 12
        add6 = pybuda.op.Add("add6", mul6, ln13)
        mul11 = pybuda.op.Multiply("mul11", ln11, ln14)
        mul12 = pybuda.op.Multiply("mul12", ln9, ln15)

        # Layer 13
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
            mul11, 
            self.gamma['gamma17'], 
            self.beta['beta17'], 
            self.dim, 
            self.epsilon
        )
        ln18 = nn.Layernorm(
            "ln18", 
            mul12, 
            self.gamma['gamma18'], 
            self.beta['beta18'], 
            self.dim, 
            self.epsilon
        )

        # Layer 14
        mul13 = pybuda.op.Multiply("mul13", ln16, ln17)
        add7 = pybuda.op.Add("add7", ln17, ln18)

        # Layer 15
        ln19 = nn.Layernorm(
            "ln19", 
            mul13, 
            self.gamma['gamma19'], 
            self.beta['beta19'], 
            self.dim, 
            self.epsilon
        )
        ln20 = nn.Layernorm(
            "ln20", 
            add7, 
            self.gamma['gamma20'], 
            self.beta['beta20'], 
            self.dim, 
            self.epsilon
        )

        # Layer 16
        add8 = pybuda.op.Add("add8", ln19, ln20)

        return add8
