# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 10
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
        super().__init__("Test 10, Layernorm")

        self.testname = "Layernorm Test 10"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = {
            f"gamma{i}": pybuda.Parameter(*self.gamma_shape, requires_grad=True)
            for i in range(1, 60)
        }
        self.beta = {
            f"beta{i}": pybuda.Parameter(*self.beta_shape, requires_grad=True)
            for i in range(1, 60)
        }
        self.train_param1 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = []
        for i in range(1, 3):
            self.set_parameter(f"train_param{i}", torch.rand(*self.input_shape, requires_grad=True))
            self.inputs.append(Tensor.create_from_torch(torch.rand(*self.input_shape)))
        for i in range(1, 60):
            self.set_parameter(f"gamma{i}", torch.rand(*self.gamma_shape, requires_grad=True))
            self.set_parameter(f"beta{i}", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        input_layer2 = [x1, self.train_param1, x2, self.train_param2]
        ln_layer2 = []
        for i in range(len(input_layer2)):
            first = 2 * i + 1
            ln_layer2.append(
                nn.Layernorm(
                    f"ln{first}",
                    input_layer2[i],
                    self.gamma[f"gamma{first}"],
                    self.beta[f"beta{first}"],
                    self.dim,
                    self.epsilon
                )
            )
            second = 2 * i + 2
            ln_layer2.append(
                nn.Layernorm(
                    f"ln{second}",
                    input_layer2[i],
                    self.gamma[f"gamma{second}"],
                    self.beta[f"beta{second}"],
                    self.dim,
                    self.epsilon
                )
            )

        # Layer 3
        ln_layer3 = []
        for i in range(5):
            first = 2 * i + 9
            ln_layer3.append(
                nn.Layernorm(
                    f"ln{first}",
                    ln_layer2[i],
                    self.gamma[f"gamma{first}"],
                    self.beta[f"beta{first}"],
                    self.dim,
                    self.epsilon
                )
            )
            second = 2 * i + 10
            ln_layer3.append(
                nn.Layernorm(
                    f"ln{second}",
                    ln_layer2[i],
                    self.gamma[f"gamma{second}"],
                    self.beta[f"beta{second}"],
                    self.dim,
                    self.epsilon
                )
            )
            
        ln19 = nn.Layernorm(
            "ln19", 
            ln_layer2[5], 
            self.gamma['gamma19'], 
            self.beta['beta19'], 
            self.dim, 
            self.epsilon
        )
        ln20 = nn.Layernorm(
            "ln20", 
            ln_layer2[6], 
            self.gamma['gamma20'], 
            self.beta['beta20'], 
            self.dim, 
            self.epsilon
        )
        ln21 = nn.Layernorm(
            "ln21", 
            ln_layer2[7], 
            self.gamma['gamma21'], 
            self.beta['beta21'], 
            self.dim, 
            self.epsilon
        )

        # Layer 4
        mul1 = pybuda.op.Multiply("mul1", ln_layer3[0], ln_layer3[2])
        mul2 = pybuda.op.Multiply("mul2", ln_layer3[1], ln_layer3[3])
        mul3 = pybuda.op.Multiply("mul3", ln_layer3[2], ln_layer3[4])
        mul4 = pybuda.op.Multiply("mul4", ln_layer3[5], ln_layer3[8])
        mul5 = pybuda.op.Multiply("mul5", ln_layer3[6], ln_layer3[9])
        add1 = pybuda.op.Add("add1", ln_layer3[7], ln20)
        add2 = pybuda.op.Add("add2", ln19, ln21)

        # Layer 5
        input_layer5 = [mul1, mul2, mul3, mul4, mul5, add1, add2]
        ln_layer5 = []
        for i in range(len(input_layer5)):
            first = 2 * i + 22
            ln_layer5.append(
                nn.Layernorm(
                    f"ln{first}",
                    input_layer5[i],
                    self.gamma[f"gamma{first}"],
                    self.beta[f"beta{first}"],
                    self.dim, 
                    self.epsilon
                )
            )
            second = 2 * i + 23
            ln_layer5.append(
                nn.Layernorm(
                    f"ln{second}",
                    input_layer5[i],
                    self.gamma[f"gamma{second}"],
                    self.beta[f"beta{second}"],
                    self.dim, 
                    self.epsilon
                )
            )

        # Layer 6
        mul6 = pybuda.op.Multiply("mul6", ln_layer5[0], ln_layer5[2])
        add3 = pybuda.op.Add("add3", ln_layer5[1], ln_layer5[3])
        mul7 = pybuda.op.Multiply("mul7", ln_layer5[4], ln_layer5[7])
        add4 = pybuda.op.Add("add4", ln_layer5[5], ln_layer5[8])
        mul8 = pybuda.op.Multiply("mul8", ln_layer5[6], ln_layer5[9])
        add5 = pybuda.op.Add("add5", ln_layer5[10], ln_layer5[12])
        mul9 = pybuda.op.Multiply("mul9", ln_layer5[11], ln_layer5[13])

        # Layer 7
        input_layer7 = [mul6, add3, mul7, add4, mul8, add5, mul9]
        ln_layer7 = []
        for i in range(len(input_layer7)):
            ln_layer7.append(
                nn.Layernorm(
                    f"ln{i + 36}",
                    input_layer7[i],
                    self.gamma[f"gamma{i + 36}"],
                    self.beta[f"beta{i + 36}"],
                    self.dim,
                    self.epsilon
                )
            )

        # Layer 8
        add6 = pybuda.op.Add("add6", ln_layer7[0], ln_layer7[1])
        add7 = pybuda.op.Add("add7", ln_layer7[2], ln_layer7[3])
        mul10 = pybuda.op.Multiply("mul10", ln_layer7[4], ln_layer7[5])
        mul11 = pybuda.op.Multiply("mul11", ln_layer7[5], ln_layer7[6])

        # Layer 9
        input_layer9 = [add6, ln_layer5[4], add7, ln_layer5[10], mul10, ln_layer5[13], mul11, ln_layer5[12]]
        ln_layer9 = []
        for i in range(len(input_layer9)):
            ln_layer9.append(
                nn.Layernorm(
                    f"ln{i + 43}",
                    input_layer9[i],
                    self.gamma[f"gamma{i + 43}"],
                    self.beta[f"beta{i + 43}"],
                    self.dim,
                    self.epsilon
                )
            )

        # Layer 10
        mul12 = pybuda.op.Multiply("mul12", ln_layer9[0], ln_layer9[2])
        mul13 = pybuda.op.Multiply("mul13", ln_layer9[1], ln_layer9[3])
        mul14 = pybuda.op.Multiply("mul14", ln_layer9[4], ln_layer9[6])
        mul15 = pybuda.op.Multiply("mul15", ln_layer9[5], ln_layer9[7])
        mul16 = pybuda.op.Multiply("mul16", ln_layer7[1], ln_layer7[4])

        # Layer 11
        input_layer11 = [mul16, mul12, mul13, mul14, mul15]
        ln_layer11 = []
        for i in range(len(input_layer11)):
            ln_layer11.append(
                nn.Layernorm(
                    f"ln{i + 51}",
                    input_layer11[i],
                    self.gamma[f"gamma{i + 51}"],
                    self.beta[f"beta{i + 51}"],
                    self.dim,
                    self.epsilon
                )
            )

        # Layer 12
        add8 = pybuda.op.Add("add8", ln_layer11[1], ln_layer11[2])
        mul17 = pybuda.op.Multiply("mul17", ln_layer9[4], ln_layer7[6])
        add9 = pybuda.op.Add("add9", ln_layer11[3], ln_layer11[4])

        # Layer 13
        ln56 = nn.Layernorm(
            "ln56", 
            add8, 
            self.gamma['gamma56'], 
            self.beta['beta56'], 
            self.dim, 
            self.epsilon
        )
        ln57 = nn.Layernorm(
            "ln57", 
            mul17, 
            self.gamma['gamma57'], 
            self.beta['beta57'], 
            self.dim, 
            self.epsilon)
        ln58 = nn.Layernorm(
            "ln58", 
            add9, 
            self.gamma['gamma58'], 
            self.beta['beta58'], 
            self.dim, 
            self.epsilon
        )

        add10 = pybuda.op.Add("add10", ln_layer11[0], ln57)

        # Layer 14
        ln59 = nn.Layernorm("ln59", add10, self.gamma['gamma59'], self.beta['beta59'], self.dim, self.epsilon)

        return ln56, ln58, ln59