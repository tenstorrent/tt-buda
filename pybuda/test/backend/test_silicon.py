# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
""" 
Basic tests of backend_api
"""

import pytest

import pybuda
from pybuda.verify import verify_module, VerifyConfig, TestKind
from pybuda.ttdevice import get_device_config
from pybuda.config import _get_global_compiler_config
from pybuda.pybudaglobal import pybuda_reset
import torch

class BudaTrain(pybuda.PyBudaModule):
    """
    Simple buda module for basic testing, with parameters
    """

    shape = (64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        in1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        in2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        sum_sqrt = pybuda.op.Sqrt("sqrt", in1)
        sum = pybuda.op.Add("add", sum_sqrt, in2)
        return sum

class BudaTest(pybuda.PyBudaModule):
    """
    Simple buda module for basic testing
    No parameters for now, to avoid using rams
    """

    shape = (64, 64)

    def __init__(self, name):
        super().__init__(name)

    def forward(self, act1, act2):
        sum = pybuda.op.Add("add", act1, act2)
        sum_sqrt = pybuda.op.Sqrt("sqrt", sum)
        return sum_sqrt   

@pytest.mark.parametrize("microbatch_size", (1, 64), ids=("microbatch1", "microbatch64"))
def test_basic_inference(test_device, microbatch_size):
    verify_module(BudaTest("verify_module"), [(microbatch_size, *BudaTest.shape), (microbatch_size, *BudaTest.shape)],
            VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch))

def get_relaxed_atol_pcc(is_training, test_device, size = 1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.89
    inference_atol = 0.1
    inference_pcc = 0.95
    if size > 1:
        training_atol = 0.5
    relative_atol = training_atol if is_training else inference_atol
    if test_device.is_silicon() and is_training:
        relative_atol *= 2.5
    pcc = training_pcc if is_training else inference_pcc

    return relative_atol, pcc


@pytest.mark.skip("Need to find the right pcc / checking to allow all the combinations")
@pytest.mark.parametrize("steps", (1, 2), ids=("s1", "s2"))
@pytest.mark.parametrize("microbatch_size", (1, 2, 64), ids=("microbatch1", "microbatch2", "microbatch64"))
@pytest.mark.parametrize("microbatch_count", (1, 4), ids=("mbc1", "mbc4"))
@pytest.mark.parametrize("accumulation_steps", (1, 2, 8), ids=("acc1", "acc2", "acc8"))
def test_basic_training(test_device, steps, microbatch_size, microbatch_count, accumulation_steps):

    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(BudaTrain("verify_module"), [(microbatch_size, *BudaTrain.shape), (microbatch_size, *BudaTrain.shape)],
            VerifyConfig(test_kind=TestKind.TRAINING, devtype=test_device.devtype, arch=test_device.arch, 
                steps=steps,
                microbatch_count=microbatch_count, 
                accumulation_steps=accumulation_steps,
                relative_atol=relative_atol,
                pcc=pcc))

def test_multi_input(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip() # input gradient data mismatch

    verify_module(BudaTest("verify_module"), [(1, *BudaTest.shape), (1, *BudaTest.shape)],
            VerifyConfig(test_kind=test_kind, microbatch_count=2, devtype=test_device.devtype, arch=test_device.arch))

@pytest.mark.parametrize("microbatch_size", (2, 64), ids=("microbatch2", "microbatch64"))
def test_microbatch(microbatch_size, test_device):
    verify_module(BudaTest("verify_module"), [(microbatch_size, 64, 64), (microbatch_size, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch))

@pytest.mark.parametrize("microbatch_size", (1, 16), ids=("microbatch1", "microbatch16"))
@pytest.mark.parametrize("microbatch_count", (1, 4), ids=("mbc1", "mbc4"))
def test_concurrent(test_kind, test_device, microbatch_size, microbatch_count):
    if test_kind.is_training():
        pytest.skip() # Runs, but there's a small input gradient error
    verify_module(BudaTest("verify_module"), [(microbatch_size, 64, 64), (microbatch_size, 64, 64)],
            VerifyConfig(
                test_kind=test_kind, 
                devtype=test_device.devtype, 
                arch=test_device.arch, 
                microbatch_count=microbatch_count,
                sequential=False))

import tensorflow as tf
from pybuda import TFModule

@pytest.mark.skip(reason="TF and fp32 problems")
def test_tf(test_device):
    class TF_Test(tf.keras.Model):
        def call(self, act1, act2):
            return act1 + act2

    verify_module(TFModule("verify_module", TF_Test()), [(1, 64, 64), (1, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch))


class MultiChipModule(pybuda.PyBudaModule):
    def __init__(self, name: str, num_devices: int):
        super().__init__(name)
        self.num_devices = num_devices
        self.weights = [pybuda.Parameter(64, 64, name = f"weights_{i}") for i in range(self.num_devices)]

    def forward(self, act):

        val = act
        for i in range(self.num_devices):
            val = pybuda.op.Matmul(f"matmul_{i}", val, self.weights[i])
            val = pybuda.op.Gelu(f"gelu_{i}", val)
            pybuda.set_chip_break(f"matmul_{i}")

        return val

def check_for_multi_chip_silicon(test_device):
    """
    Skip the test if there's only one (or less) silion devices available. Return number of devices.
    """
    if test_device.devtype == pybuda.BackendType.Golden:
        pytest.skip("Not meant for golden")

    num_devices = len(pybuda.detect_available_devices())
    if num_devices < 2:
        pytest.skip("Need at least 2 devices to run multi-chip test")

    return num_devices


def test_multi_chip(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip("Training on multiple chips not working yet")

    num_devices = check_for_multi_chip_silicon(test_device)

    microbatch_size = 1
    verify_module(MultiChipModule("multi_chip_module", num_devices), [(microbatch_size, 64, 64)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, chip_ids=list(range(num_devices))), 
            params_centered_on_zero=True, inputs_centered_on_zero=True)

def test_chip_id(test_kind, test_device):

    if test_device.devtype == pybuda.BackendType.Golden:
        pytest.skip("Not meant for golden")

    num_devices = check_for_multi_chip_silicon(test_device)

    if num_devices < 2:
        pytest.skip("Need at least 2 devices to run chip-id test")

    # Run on last device
    microbatch_size = 4
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind.is_training(), test_device)
    verify_module(BudaTrain("last_device"), [(microbatch_size, 64, 64), (microbatch_size, 64, 64)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, chip_ids=[num_devices-1],
                relative_atol=relative_atol, pcc=pcc),
            params_centered_on_zero=False, inputs_centered_on_zero=False)
