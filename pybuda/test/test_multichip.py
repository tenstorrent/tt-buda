# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

import os
import pybuda
import pybuda.op
from pybuda import (
    Tensor,
    CompilerConfig,
)
from pybuda.verify import verify_module, VerifyConfig, TestKind
from pybuda._C.backend_api import BackendDevice
from .common import compile, device, ModuleBuilder
from pybuda.config import _get_global_compiler_config

backend_devices = {
    "grayskull" : BackendDevice.Grayskull,
    "wormhole_b0": BackendDevice.Wormhole_B0,
}

# Currently only guarded for Grayskull:
#  - DRAM Input queues for consuming ops are expected to be on the same local device
#  - For cases where a remote producer must feed multiple remote chips, we pass the
#    data through a nop indirection for each consumer.
#  - For cases where the input-queue being read/consumed is a constant, we replicate
#    the data to be local per chip

def test_multichip_input_queue_forks_to_multiple_remote_chips():
    shape = (1, 1, 64, 64)

    arch = backend_devices[os.environ.get("BACKEND_ARCH_NAME", "grayskull")]

    compiler_cfg = CompilerConfig(enable_training=False)
    compiler_cfg.enable_consteval = False
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False if arch is BackendDevice.Grayskull else True

    @compile(
        compiler_cfg=compiler_cfg,
        verify_cfg=VerifyConfig(arch=arch),
        chip_ids=[0, 1, 2, 3]
    )
    def three_branch_input_queue(act):
        branch_a = pybuda.op.Gelu(f"branch_a", act)
        branch_b = pybuda.op.Gelu(f"branch_b", act)
        branch_c = pybuda.op.Gelu(f"branch_c", act)
        add_a = pybuda.op.Add("add_a", branch_b, branch_a)
        add_b = pybuda.op.Add("add_b", add_a, branch_c)
        return add_b

    act = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    compilation_results = three_branch_input_queue(act)
    placer_solution = compilation_results.pass_specific_output_kwargs["placer_solution"]



@pytest.mark.parametrize("manual_placement", [True, False])
def test_multichip_producer_forks_to_multiple_remote_chips(manual_placement):
    shape = (1, 1, 64, 64)

    arch = backend_devices[os.environ.get("BACKEND_ARCH_NAME", "grayskull")]

    compiler_cfg = CompilerConfig(enable_training=False)
    compiler_cfg.enable_consteval = False
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False if arch is BackendDevice.Grayskull else True
    if manual_placement:
        compiler_cfg.place_on_new_chip("branch_a")
        compiler_cfg.place_on_new_chip("branch_b")
        compiler_cfg.place_on_new_chip("branch_c")

    @compile(
        compiler_cfg=compiler_cfg,
        verify_cfg=VerifyConfig(arch=arch),
        chip_ids=[0, 1, 2, 3]
    )
    def three_branch_fork(act):
        nop = pybuda.op.Buffer(f"nop", act)
        fork = pybuda.op.Buffer(f"fork", nop)

        branch_a = pybuda.op.Gelu(f"branch_a", fork)
        branch_b = pybuda.op.Gelu(f"branch_b", fork)
        branch_c = pybuda.op.Gelu(f"branch_c", fork)
        add_a = pybuda.op.Add("add_a", branch_b, branch_a)
        add_b = pybuda.op.Add("add_b", add_a, branch_c)
        return add_b

    act = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    compilation_results = three_branch_fork(act)
    placer_solution = compilation_results.pass_specific_output_kwargs["placer_solution"]


def test_multichip_constant_forks_to_multiple_remote_chips():
    shape = (1, 1, 64, 64)

    compiler_cfg = CompilerConfig(enable_training=False)
    compiler_cfg.place_on_new_chip("constant_consumer_A")
    compiler_cfg.place_on_new_chip("constant_consumer_B")
    compiler_cfg.enable_consteval = False
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False

    @compile(
        compiler_cfg=compiler_cfg,
        verify_cfg=VerifyConfig(arch=BackendDevice.Grayskull),
        chip_ids=[0, 1, 2]
    )
    def constant_two_branch_fork(act):
        constant = pybuda.op.Constant("constant", constant=1.0)

        left_branch = pybuda.op.Buffer(f"constant_consumer_A", constant)
        right_branch = pybuda.op.Buffer(f"constant_consumer_B", constant)
        add = pybuda.op.Add("add_consumers", left_branch, right_branch)
        final_add = pybuda.op.Add("add", act, add)
        return final_add

    act = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    compilation_results = constant_two_branch_fork(act)
    placer_solution = compilation_results.pass_specific_output_kwargs["placer_solution"]

    assert placer_solution.chip_id("constant_consumer_A") != placer_solution.chip_id("constant_consumer_B")

def test_multichip_wormhole_sanity():
    def linked_list_two_chips(act):
        op0 = pybuda.op.Gelu(f"op0", act)
        op1 = pybuda.op.Gelu(f"op1", op0)
        return op1

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False

    module = ModuleBuilder(linked_list_two_chips)
    verify_module(module, [(1, 1, 64, 64)],
            # chip_ids=[0, 1] fails in net2pipe bbe_issue#2331
            # VerifyConfig(test_kind=TestKind.INFERENCE, run_net2pipe=True, arch=BackendDevice.Wormhole_B0, chip_ids=[0,1]))
            VerifyConfig(test_kind=TestKind.INFERENCE, run_net2pipe=True))

def test_four_chip_wormhole_sanity():
    pytest.skip("Skip until BBE commit 42d9685b1 is consumed")
    def linked_list_four_chips(act):
        op0 = pybuda.op.Gelu(f"op0", act)
        op1 = pybuda.op.Gelu(f"op1", op0)
        return op1

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False
    pybuda.set_configuration_options(
        backend_cluster_descriptor_path="third_party/budabackend/wormhole_2x4_sequential_cluster.yaml"
    )

    module = ModuleBuilder(linked_list_four_chips)
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, run_net2pipe=True, arch=BackendDevice.Wormhole_B0, chip_ids=list(range(8))))



@pytest.mark.skip("Skip until #736 is solved")
def test_linked_list_multichip_auto_placer():
    shape = (1, 1, 64, 64)

    compiler_cfg = CompilerConfig(enable_training=False)
    compiler_cfg.enable_consteval = False
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False

    @compile(
        compiler_cfg=compiler_cfg,
        verify_cfg=VerifyConfig(arch=BackendDevice.Grayskull),
        chip_ids=[0, 1, 2, 3]
    )
    def linked_list(act):
        a_out = pybuda.op.Buffer(f"A", act)
        b_out = pybuda.op.Buffer(f"B", a_out)
        c_out = pybuda.op.Buffer(f"C", b_out)
        d_out = pybuda.op.Buffer(f"D", c_out)
        return d_out

    act = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    compilation_results = linked_list(act)
    placer_solution = compilation_results.pass_specific_output_kwargs["placer_solution"]

    assert placer_solution.chip_id("A") == 0
    assert placer_solution.chip_id("B") == 1
    assert placer_solution.chip_id("C") == 2
    assert placer_solution.chip_id("D") == 3



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


def test_training_sanity_multichip_grayskull(test_device):
    microbatch_size = 1
    pybuda.set_chip_break("sqrt")
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False
    compiler_cfg.use_interactive_placer = False
    verify_module(BudaTrain("verify_module"), [(microbatch_size, *BudaTrain.shape), (microbatch_size, *BudaTrain.shape)],
            VerifyConfig(test_kind=TestKind.TRAINING, arch=BackendDevice.Grayskull, 
                steps=1,
                microbatch_count=1,
                accumulation_steps=1,
                chip_ids=list(range(2))))

def test_training_sanity_multichip_wormhole(test_device):
    microbatch_size = 1
    pybuda.set_chip_break("sqrt")
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False
    verify_module(BudaTrain("verify_module"), [(microbatch_size, *BudaTrain.shape), (microbatch_size, *BudaTrain.shape)],
            VerifyConfig(test_kind=TestKind.TRAINING, arch=BackendDevice.Wormhole_B0, 
                steps=1,
                microbatch_count=1,
                accumulation_steps=1,
                chip_ids=list(range(2))))