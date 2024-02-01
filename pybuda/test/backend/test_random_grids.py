# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import pytest
from pybuda.verify import verify_module, VerifyConfig

microbatch_size = 8

class MatmulSimple(pybuda.PyBudaModule):
    shape = (256, 256)
    def __init__(self, name: str):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        return pybuda.op.Add("add", m1, m2)

class MatmulDramFork(pybuda.PyBudaModule):
    shape = (256, 256)
    def __init__(self, name: str):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act1, self.weights2)
        add = pybuda.op.Add("add", m1, m2)
        return pybuda.op.Add("add_final", add, act2)

class EltwiseFork(pybuda.PyBudaModule):
    shape = (256, 256)
    def __init__(self, name: str):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        add = pybuda.op.Add("first_add", act1, act2)
        m1 = pybuda.op.Matmul("matmul1", add, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", add, self.weights2)
        return pybuda.op.Add("add", m1, m2)

class DoubleFork(pybuda.PyBudaModule):
    shape = (256, 256)
    def __init__(self, name: str):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        add = pybuda.op.Add("first_add", act1, act2)
        weight_add = pybuda.op.Add("weight_add", self.weights1, self.weights2)
        m1 = pybuda.op.Matmul("matmul1", add, weight_add)
        m2 = pybuda.op.Matmul("matmul2", add, weight_add)
        return pybuda.op.Add("add", m1, m2)

@pytest.mark.parametrize("model", [MatmulSimple, MatmulDramFork, EltwiseFork, DoubleFork])
def test(test_kind, test_device, model):
    pybuda.set_configuration_options(balancer_policy="Random")

    verify_module(model("random_grid"), [(microbatch_size, *model.shape), (microbatch_size, *model.shape)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch))


