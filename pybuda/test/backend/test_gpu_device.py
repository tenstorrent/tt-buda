# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test combinations of front-end modules on each available device type
"""

import pytest

import torch

import pybuda
from pybuda import PyTorchModule
from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig, TestKind

from pybuda._C.backend_api import BackendType

import nvidia_smi

def does_gpu_have_enough_available_ram():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    # If more than 2GBs of RAM are used, then GPU doesn't have enough RAM  (number was picked arbitrarily)
    has_enough_ram =  True
    if info.used  > 3  * 1024  * 1024 * 1024:
        has_enough_ram = False

    nvidia_smi.nvmlShutdown()

    return has_enough_ram

GPU_HAS_ENOUGH_RAM =  does_gpu_have_enough_available_ram()

# https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
mp_context = torch.multiprocessing.get_context('spawn')

class PytorchUnary(torch.nn.Module):
    def forward(self, x):
        return 1 - x

def test_pytorch_gpu(test_kind):
    if not GPU_HAS_ENOUGH_RAM:
        pytest.skip("GPU doesn't have enough RAM")

    if not torch.cuda.is_available():
        pytest.skip("Pytorch didn't detect cuda")

    verify_module(PyTorchModule("test", PytorchUnary()), [(1, 1, 64, 64)], 
            VerifyConfig(test_kind=test_kind, pcc=0.99), device_type="GPUDevice")


class MyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64, bias=True)

    def forward(self, act):
        x = self.lin(act)
        return x

def test_pt_linear_pipeline():
    if not GPU_HAS_ENOUGH_RAM:
        pytest.skip("GPU doesn't have enough RAM")
        
    if not torch.cuda.is_available():
        pytest.skip("Pytorch didn't detect cuda")

    verify_module(PyTorchModule("pt_linear", MyLinear().cuda()), [(1, 128, 64)], 
            VerifyConfig(test_kind=TestKind.INFERENCE, pcc=0.99), device_type="GPUDevice")

# Sample PyBuda module
class PyBudaTestModule(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = pybuda.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        return m1 + m2, m2

# Sample PyTorch module
class PyTorchTestModuleOneOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights1 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = torch.matmul(act1, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2

# Run training pipeline, with loss on CPU, read back checkpoints and loss
def test_training_pipeline_read_back():
    if not GPU_HAS_ENOUGH_RAM:
        pytest.skip("GPU doesn't have enough RAM")

    if not torch.cuda.is_available():
        pytest.skip("Pytorch didn't detect cuda")

    tt0 = pybuda.TTDevice("tt0", module=PyBudaTestModule("stage0"), devtype=BackendType.Golden)
    cpu1 = pybuda.GPUDevice("gpu1", module=pybuda.PyTorchModule("stage1", PyTorchTestModuleOneOut().cuda()))
    cpu1.place_loss_module(pybuda.PyTorchModule("l1loss", torch.nn.L1Loss().cuda()))

    import torch.multiprocessing as mp
    loss_q = mp_context.Queue()
    checkpoint_q = mp_context.Queue()

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0.push_to_inputs((input1, input2))

    cpu1.push_to_target_inputs(torch.rand(4, 32, 32))

    pybuda.run_training(checkpoint_queue = checkpoint_q, loss_queue=loss_q)

    print("checkpoint: ", checkpoint_q.get())
    print("loss: ", loss_q.get())
