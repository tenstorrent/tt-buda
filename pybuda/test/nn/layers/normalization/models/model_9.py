# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 9
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
        super().__init__("Test 9, Layernorm")

        self.testname = "Layernorm Test 9"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = {
            f"gamma{i}": pybuda.Parameter(*self.gamma_shape, requires_grad=True)
            for i in range(1, 57)
        }
        self.beta = {
            f"beta{i}": pybuda.Parameter(*self.beta_shape, requires_grad=True)
            for i in range(1, 57)
        }
        self.train_param1 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.input_shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = []
        for i in range(1, 4):
            self.set_parameter(f"train_param{i}", torch.rand(*self.input_shape, requires_grad=True))
            self.inputs.append(Tensor.create_from_torch(torch.rand(*self.input_shape)))
        for i in range(1, 57):
            self.set_parameter(f"gamma{i}", torch.rand(*self.gamma_shape, requires_grad=True))
            self.set_parameter(f"beta{i}", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        ln_layer2 = []
        activations = [x1, x2, x3]
        for i in range(3):
            first = 2 * i + 1
            ln_layer2.append(
                nn.Layernorm(
                    f"ln{first}", 
                    activations[i], 
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
                    activations[i], 
                    self.gamma[f"gamma{second}"], 
                    self.beta[f"beta{second}"], 
                    self.dim, 
                    self.epsilon
                )
            )

        # Layer 3
        ln_layer3 = []
        for i in range(6):
            first = 2 * i + 7
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
            second = 2 * i + 8
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

        # Layer 4
        mul1 = pybuda.op.Multiply("mul1", ln_layer3[0], ln_layer3[2])
        mul2 = pybuda.op.Multiply("mul2", ln_layer3[1], ln_layer3[3])
        mul3 = pybuda.op.Multiply("mul3", ln_layer3[4], ln_layer3[6])
        mul4 = pybuda.op.Multiply("mul4", ln_layer3[5], ln_layer3[7])
        mul5 = pybuda.op.Multiply("mul5", ln_layer3[8], ln_layer3[10])
        mul6 = pybuda.op.Multiply("mul6", ln_layer3[9], ln_layer3[11])

        # Layer 5
        input_layer5 = [mul1, mul2, mul3, mul4, mul5, mul6]
        ln_layer5 = []
        for i in range(6):
            first = 2 * i + 19
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
            second = 2 * i + 20
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
        mul7 = pybuda.op.Multiply("mul7", ln_layer5[2], self.train_param1)
        mul8 = pybuda.op.Multiply("mul8", ln_layer5[6], self.train_param2)
        mul9 = pybuda.op.Multiply("mul9", ln_layer5[10], self.train_param3)

        # Layer 7
        ln31 = nn.Layernorm(
            "ln31", 
            mul7, 
            self.gamma['gamma31'], 
            self.beta['beta31'], 
            self.dim, 
            self.epsilon
        )
        ln32 = nn.Layernorm(
            "ln32", 
            mul8, 
            self.gamma['gamma32'], 
            self.beta['beta32'], 
            self.dim, 
            self.epsilon
        )
        ln33 = nn.Layernorm(
            "ln33", 
            mul9, 
            self.gamma['gamma33'], 
            self.beta['beta33'], 
            self.dim, 
            self.epsilon
        )

        # Layer 8
        add1 = pybuda.op.Add("add1", ln_layer5[1], ln31)
        add2 = pybuda.op.Add("add2", ln_layer5[4], ln32)
        add3 = pybuda.op.Add("add3", ln_layer5[8], ln33)

        # Layer 9
        mul10 = pybuda.op.Multiply("mul10", ln_layer5[0], add1)
        mul11 = pybuda.op.Multiply("mul11", ln_layer5[3], add2)
        add4 = pybuda.op.Add("add4", ln_layer5[5], ln32)
        add5 = pybuda.op.Add("add5", ln_layer5[7], add3)
        mul12 = pybuda.op.Multiply("mul12", ln_layer5[9], ln33)
        mul13 = pybuda.op.Multiply("mul13", mul9, ln_layer5[11])
        
        # Layer 10
        input_layer10 = [mul10, mul11, add4, add5, mul12, mul13]
        ln_layer10 = []
        for i in range(6):
            first = 2 * i + 34
            ln_layer10.append(
                nn.Layernorm(
                    f"ln{first}",
                    input_layer10[i],
                    self.gamma[f"gamma{first}"],
                    self.beta[f"beta{first}"],
                    self.dim, 
                    self.epsilon
                )
            )
            second = 2 * i + 35
            ln_layer10.append(
                nn.Layernorm(
                    f"ln{second}",
                    input_layer10[i], 
                    self.gamma[f"gamma{second}"],
                    self.beta[f"beta{second}"],
                    self.dim, 
                    self.epsilon
                )
            )

        # Layer 11
        add6 = pybuda.op.Add("add6", ln_layer10[0], ln_layer10[1])
        add7 = pybuda.op.Add("add7", ln_layer10[2], ln_layer10[3])
        add8 = pybuda.op.Add("add8", ln_layer10[4], ln_layer10[5])
        add9 = pybuda.op.Add("add9", ln_layer10[6], ln_layer10[7])
        add10 = pybuda.op.Add("add10", ln_layer10[8], ln_layer10[9])
        add11 = pybuda.op.Add("add11", ln_layer10[10], ln_layer10[11])

        # Layer 12
        input_layer12 = [add6, add7, add8, add9, add10, add11]
        ln_layer12 = []
        for i in range(6):
            ln_layer12.append(
                nn.Layernorm(
                    f"ln{i + 46}",
                    input_layer12[i], 
                    self.gamma[f"gamma{i + 46}"],
                    self.beta[f"beta{i + 46}"],
                    self.dim, 
                    self.epsilon
                )
            )

        # Layer 13
        add12 = pybuda.op.Add("add12", ln_layer12[0], ln_layer12[2])
        add13 = pybuda.op.Add("add13", ln_layer12[1], ln_layer12[4])
        add14 = pybuda.op.Add("add14", ln_layer12[3], ln_layer12[5])

        # Layer 14
        ln52 = nn.Layernorm(
            "ln52", 
            add12, 
            self.gamma['gamma52'], 
            self.beta['beta52'], 
            self.dim, 
            self.epsilon
        )
        ln53 = nn.Layernorm(
            "ln53", 
            add13, 
            self.gamma['gamma53'], 
            self.beta['beta53'], 
            self.dim, 
            self.epsilon
        )
        ln54 = nn.Layernorm(
            "ln54", 
            add14, 
            self.gamma['gamma54'], 
            self.beta['beta54'], 
            self.dim, 
            self.epsilon
        )

        # Layer 15
        mul14 = pybuda.op.Multiply("mul14", ln52, ln53)
        mul15 = pybuda.op.Multiply("mul15", ln53, ln54)

        # Layer 16
        ln55 = nn.Layernorm(
            "ln55", 
            mul14, 
            self.gamma['gamma55'], 
            self.beta['beta55'], 
            self.dim, 
            self.epsilon
        )
        ln56 = nn.Layernorm(
            "ln56", 
            mul15, 
            self.gamma['gamma56'], 
            self.beta['beta56'], 
            self.dim, 
            self.epsilon
        )

        return ln55, ln56