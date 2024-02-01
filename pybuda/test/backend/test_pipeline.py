# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline tests for different combinations of CPU / TT devices (and eventually GPU)
"""

import torch
import pytest
import tensorflow as tf

import pybuda
from pybuda import PyTorchModule, BackendType, TFModule
from pybuda.verify import verify_module_pipeline, VerifyConfig, TestKind

class PytorchUnary(torch.nn.Module):
    def forward(self, x):
        return 1.5 - x


def test_module_pipeline_cpu(test_kind):
    verify_module_pipeline([PyTorchModule("pipe0", PytorchUnary()), PyTorchModule("pipe1", PytorchUnary())], [(1, 64, 64)], 
            VerifyConfig(test_kind=test_kind), device_types=["CPUDevice"])

def test_module_pipeline_tt():
    verify_module_pipeline([PyTorchModule("pipe0", PytorchUnary()), PyTorchModule("pipe1", PytorchUnary())], [(1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE), device_types=["TTDevice"])


@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_device_pipeline_2(first, second):
    verify_module_pipeline([PyTorchModule("pipe0", PytorchUnary()), PyTorchModule("pipe1", PytorchUnary())], [(1, 32, 32)], 
            VerifyConfig(test_kind=TestKind.INFERENCE), device_types=[first, second])

#@pytest.mark.skip(reason="Not passing yet")
@pytest.mark.parametrize("third", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_device_pipeline_3(first, second, third):
    verify_module_pipeline(
            [PyTorchModule("pipe0", PytorchUnary()), 
             PyTorchModule("pipe1", PytorchUnary()),
             PyTorchModule("pipe2", PytorchUnary()),
            ], [(1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE), device_types=[first, second, third])


class PytorchMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(32, 32) - 0.5, requires_grad=True)

    def forward(self, act):
        act = act.type(torch.float32) # TODO: make this somewhat automatic?
        return torch.matmul(act, self.weights)

class PybudaMatmul(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(32, 32, requires_grad=True)

    def forward(self, act):
        return pybuda.op.Matmul(self.name + ".matmul", act, self.weights)

@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_training_pipeline_1(first):
    verify_module_pipeline([PyTorchModule("pipe0", PytorchMatmul())], [(1, 32, 32)], 
            VerifyConfig(test_kind=TestKind.TRAINING, relative_atol=30), device_types=[first])

@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_training_pipeline_2(first, second):
    verify_module_pipeline([PyTorchModule("pipe0", PytorchMatmul()), PyTorchModule("pipe1", PytorchMatmul())], [(1, 32, 32)], 
            VerifyConfig(test_kind=TestKind.TRAINING), device_types=[first, second])

@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_training_pipeline_2_pybuda(test_device, first, second):
    #sequential = test_device.devtype == BackendType.Golden
    sequential = True
    verify_module_pipeline([
        PyTorchModule("pipe0", PytorchMatmul()) if first == "CPUDevice" else PybudaMatmul("pipe0"), 
        PyTorchModule("pipe1", PytorchMatmul()) if second == "CPUDevice" else PybudaMatmul("pipe1"), 
        ], [(1, 32, 32)], 
            VerifyConfig(
                test_kind=TestKind.TRAINING, arch=test_device.arch, devtype=test_device.devtype, sequential=sequential), 
            device_types=[first, second])

@pytest.mark.parametrize("third", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_training_pipeline_3(first, second, third):
    verify_module_pipeline([
        PyTorchModule("pipe0", PytorchMatmul()), 
        PyTorchModule("pipe1", PytorchMatmul()),
        PyTorchModule("pipe2", PytorchMatmul()),
        ], [(1, 32, 32)], 
            VerifyConfig(test_kind=TestKind.TRAINING, relative_atol=0.35), device_types=[first, second, third])

@pytest.mark.parametrize("third", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_training_pipeline_3_pybuda(first, second, third):
    verify_module_pipeline([
        PyTorchModule("pipe0", PytorchMatmul()) if first == "CPUDevice" else PybudaMatmul("pipe0"), 
        PyTorchModule("pipe1", PytorchMatmul()) if second == "CPUDevice" else PybudaMatmul("pipe1"), 
        PyTorchModule("pipe2", PytorchMatmul()) if third == "CPUDevice" else PybudaMatmul("pipe2"), 
        ], [(1, 32, 32)], 
            VerifyConfig(test_kind=TestKind.TRAINING, relative_atol=0.35), device_types=[first, second, third])


@pytest.mark.parametrize("second", ("CPUDevice", "TTDevice"))
@pytest.mark.parametrize("first", ("CPUDevice", "TTDevice"))
def test_tf_matmul_pipeline(test_device, first, second):
    class Linear1(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(64, use_bias=False)

        def call(self, x1):
            return self.dense1(x1)

    class Linear2(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense2 = tf.keras.layers.Dense(64, use_bias=False)

        def call(self, x1):
            return self.dense2(x1)

    l1 = TFModule("l1", Linear1())
    l2 = TFModule("l2", Linear2())

    input_tensor = tf.Variable(tf.random.uniform((1, 64, 64)))
    verify_module_pipeline([l1, l2], [(1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.TRAINING, relative_atol=0.35,
            devtype=test_device.devtype, arch=test_device.arch,), device_types=[first, second],inputs=[(input_tensor, ), ],)


def test_tf_matmul_multiple_cpu_module(test_device):
    class Linear1(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(64, use_bias=False)

        def call(self, x1):
            return self.dense1(x1)

    class Linear2(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense2 = tf.keras.layers.Dense(64, use_bias=False)

        def call(self, x1):
            return self.dense2(x1)

    l1 = TFModule("l1", Linear1())
    l2 = TFModule("l2", Linear2())

    input_tensor = tf.Variable(tf.random.uniform((1, 64, 64)))
    verify_module_pipeline([l1, l2], [(1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.TRAINING, relative_atol=0.35,
            devtype=test_device.devtype, arch=test_device.arch,), device_types=["CPUDevice"],inputs=[(input_tensor, ), ],)
