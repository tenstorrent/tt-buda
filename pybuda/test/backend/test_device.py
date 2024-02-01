# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test combinations of front-end modules on each available device type
"""

import pytest

import torch
import tensorflow as tf

import pybuda
from pybuda import PyTorchModule, TFModule, PyBudaModule
from pybuda.verify import verify_module, VerifyConfig, TestKind

class PytorchUnary(torch.nn.Module):
    def forward(self, x):
        return 1 - x

class TFUnary(tf.keras.Model):
    def call(self, x):
        return 1 - x

class PyBudaUnary(PyBudaModule):
    def forward(self, x):
        return 1 - x

def test_pytorch_cpu():
    verify_module(PyTorchModule("test", PytorchUnary()), [(1, 1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="CPUDevice")

def test_pytorch_tt():
    verify_module(PyTorchModule("test", PytorchUnary()), [(1, 1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="TTDevice")

def test_tf_cpu():
    verify_module(TFModule("test", TFUnary()), [(1, 1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="CPUDevice")

def test_tf_tt():
    verify_module(TFModule("test", TFUnary()), [(1, 1, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="TTDevice")

def test_pybuda_tt():
    verify_module(PyBudaUnary("test"), [(1, 1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="TTDevice")

@pytest.mark.skip(reason="Not supported yet")
def test_pybuda_cpu(): 
    verify_module(PyBudaUnary("test"), [(1, 1, 64, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="CPUDevice")

#def test_mix_tt():
#    verify_module_pipeline([PyTorchModule("pipe0", PytorchUnary()), PyTorchModule("pipe1", PytorchUnary())], [(1, 1, 64, 64)], 
#            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type=CPUDevice)
