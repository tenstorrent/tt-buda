# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
""" 
Basic tests of backend_api
"""

import pytest

import pybuda
from pybuda import DataFormat, BackendDevice, BackendType
from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig, TestKind

class BudaTest(pybuda.PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (64, 64)

    def __init__(self, name: str, multi_output: bool = False):
        super().__init__(name)
        self.multi_output = multi_output
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        m1e = pybuda.op.Sqrt("sqrt", m1)
        add = pybuda.op.Add("add", m1e, m2)
        if self.multi_output:
            return m1e, add
        else:
            return add

def test_basic(test_kind):
    verify_module(BudaTest("verify_module"), [(2, 64, 64), (2, 64, 64)],
            VerifyConfig(test_kind=test_kind, run_net2pipe=True))

def test_basic_wormhole(test_kind):
    if test_kind == TestKind.TRAINING_RECOMPUTE:
        pytest.skip() # tenstorrent/budabackend#382
    verify_module(BudaTest("verify_module"), [(1, 64, 64), (1, 64, 64)],
           VerifyConfig(test_kind=test_kind, run_net2pipe=True))

def test_multi_input_inference():
    verify_module(BudaTest("verify_module"), [(1, 64, 64), (1, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, run_net2pipe=True, microbatch_count=10))

def test_multi_output(test_kind):
    verify_module(BudaTest("verify_module", multi_output=True), [(1, 64, 64), (1, 64, 64)],
            VerifyConfig(test_kind=test_kind, run_net2pipe=True))


@pytest.mark.parametrize("steps", (1, 2), ids=("s1", "s2"))
@pytest.mark.parametrize("accumulation_steps", (1, 2), ids=("as1", "as2"))
@pytest.mark.parametrize("microbatch_count", (1, 2), ids=("mbc1", "mbc2"))
@pytest.mark.parametrize("microbatch_size", (1, 2, 16), ids=("mb1", "mb2", "mb16"))
def test_multi_input_training(steps, accumulation_steps, microbatch_count, microbatch_size):
    #if steps > 1 or accumulation_steps > 1 or microbatch_size > 1 or microbatch_count > 1:
    #    pytest.skip() # skip while debugging

    acc_step_inputs = microbatch_size * microbatch_count
    pcc = None
    if acc_step_inputs >= 32:
        pcc = 0.93
    elif acc_step_inputs >= 16:
        pcc = 0.97

    verify_module(BudaTest("verify_module"), [(microbatch_size, 64, 64), (microbatch_size, 64, 64)],
            VerifyConfig(
                test_kind=TestKind.TRAINING, 
                steps=steps,
                pcc=pcc,
                accumulation_steps=accumulation_steps,
                microbatch_count=microbatch_count,
                run_net2pipe=True,
                scale_loss=1.0))

@pytest.mark.parametrize("microbatch_size", (2, 64), ids=("microbatch2", "microbatch64"))
def test_microbatch_inference(microbatch_size):
    verify_module(BudaTest("verify_module"), [(microbatch_size, 64, 64), (microbatch_size, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, run_net2pipe=True))

@pytest.mark.skip(reason="Golden doesn't support concurrent")
def test_concurrent(test_kind):
    verify_module(BudaTest("verify_module"), [(64, 64), (64, 64)],
            VerifyConfig(test_kind=test_kind, sequential=False, run_net2pipe=True))

# Test the scenario where tile broadcast folds into an input node
class InputFolding(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.eltwise_param = pybuda.Parameter(64, 64, requires_grad=True)

    def forward(self, act):

        # (1, 1) + (64, 64) - need to scalar-broadcast act to get correct result
        add = pybuda.op.Add("add", act, self.eltwise_param)
        return add

def test_input_folding(test_kind): 
    if test_kind.is_training():
        pytest.skip() # slight gradient error, need to debug to see if it's real
    verify_module(InputFolding("backend_input_folding"), [(1, 1)],
            VerifyConfig(test_kind=test_kind, sequential=True, run_net2pipe=True))


# Test a simple PT model
from pybuda import PyTorchModule
import torch

class MyLinearNoConstEval(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 128, bias=True)
        self.lin.weight = torch.nn.Parameter(self.lin.weight.transpose(-2, -1))

    def forward(self, act):
        x = torch.matmul(act, self.lin.weight)
        x = x + self.lin.bias
        return x

class MyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 128, bias=True)

    def forward(self, act):
        x = self.lin(act)
        return x

def test_pt_linear(test_kind):
    verify_module(PyTorchModule("pt_linear", MyLinear()), [(1, 128, 64)],
            VerifyConfig(test_kind=test_kind, sequential=True, relative_atol=0.25))

class MyLinearCPU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64, bias=True)

    def forward(self, act):
        x = self.lin(act)
        return x

def test_pt_linear_pipeline(test_kind):
    dev_model = PyTorchModule("pt_linear", MyLinear())
    cpu_model = PyTorchModule("pt_linear", MyLinearCPU())
    verify_module_pipeline([cpu_model, dev_model], 
            [(1, 128, 64)],
            VerifyConfig(test_kind=test_kind, accumulation_steps=1, relative_atol=0.25),
            device_types=["CPUDevice", "TTDevice"]
    )

def test_pt_linear_pipeline_no_consteval(test_kind):
    dev_model = PyTorchModule("pt_linear_no_ce", MyLinearNoConstEval())
    cpu_model = PyTorchModule("pt_linear", MyLinearCPU())
    verify_module_pipeline([cpu_model, dev_model], 
            [(1, 128, 64)],
            VerifyConfig(test_kind=test_kind, accumulation_steps=1, relative_atol=0.25),
            device_types=["CPUDevice", "TTDevice"]
    )

class LargeParam(pybuda.PyBudaModule):
    def __init__(self, name: str):
        super().__init__(name)
        self.weights = pybuda.Parameter(torch.rand(2048, 2048))

    def forward(self, act):
        return pybuda.op.Matmul("matmul", act, self.weights)


def test_memory_leak_parameter_gradients():
    pytest.skip("Random fails in CI due to other processes using memory")

    tt0 = pybuda.TTDevice("tt0", module=LargeParam("large_param"), arch=BackendDevice.Grayskull, devtype=BackendType.Golden)
    tt0.place_loss_module(pybuda.op.loss.L1Loss("l1_loss"))
    inputs = torch.rand(1, 64, 2048)
    target = torch.rand(1, 64, 2048)

    tt0.push_to_inputs((inputs,))
    tt0.push_to_target_inputs((target,))
    pybuda.run_training()

    print("Reading gradients first time")
    pybuda.get_parameter_gradients(tt0, _sequential=True)
    import psutil
    first_mem_use = psutil.virtual_memory().used / (1024*1024)

    for i in range(100):
        print(f"Reading gradients ({i+1}/100)")
        pybuda.get_parameter_gradients(tt0, _sequential=True)

    final_mem_use = psutil.virtual_memory().used / (1024 * 1024)

    print(first_mem_use, final_mem_use, final_mem_use - first_mem_use)
    assert final_mem_use - first_mem_use < 150, "More than 150MB leaked"
    
