# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import os

import pybuda
import pybuda.op
from pybuda import (
    PyBudaModule,
    TTDevice,
    BackendType,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
    CompileDepth,
)
from pybuda._C.backend_api import BackendDevice
from .common import compile, device, ModuleBuilder, run
import pybuda.verify as verify
import pybuda.query as query


verify_cfg = VerifyConfig(run_golden=True) # Run backend golden check on all tests in here

class BudaTest(PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        m1e = pybuda.op.Exp("exp", m1)
        return pybuda.op.Add("add", m1e, m2)


class ForkJoinTest(PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act):
        nop = pybuda.op.Buffer("nop", act)
        m1 = pybuda.op.Matmul("matmul1", nop, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", nop, self.weights2)
        m1_nop = pybuda.op.Buffer("matmul1_nop", m1)
        m2_nop = pybuda.op.Buffer("matmul2_nop", m2)
        return pybuda.op.Add("add", m1_nop, m2_nop)


def test_epoch_break():
    training = False

    mod = BudaTest("test_module")
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))
    act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))
    mod.set_parameter("weights2", torch.rand(*BudaTest.shape, requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    compiler_cfg = CompilerConfig(enable_training=training, compile_depth=CompileDepth.BALANCER_PASS)
    compiler_cfg.place_on_new_epoch("matmul2")
    compiler_cfg.place_on_new_epoch("exp")
    compiler_cfg.place_on_new_epoch("add")

    compile_result = pybuda_compile(tt0, "sanity", act1, act2, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    epochs = set()
    for op in ["matmul1", "matmul2", "exp", "add"]:
        epoch_id = placer_solution.epoch_id(op)
        assert op not in epochs
        epochs.add(epoch_id)

def test_chip_break():
    training = False

    mod = BudaTest("test_module")
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer, chip_ids=[0, 1, 2])
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))
    act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))
    mod.set_parameter("weights2", torch.rand(*BudaTest.shape, requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    compiler_cfg = CompilerConfig(enable_training=training, compile_depth=CompileDepth.BALANCER_PASS)
    compiler_cfg.place_on_new_chip("exp")
    compiler_cfg.place_on_new_chip("add")
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False

    compile_result = pybuda_compile(tt0, "sanity", act1, act2, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    assert placer_solution.epoch_id("exp") - placer_solution.epoch_id("matmul1") == 1
    assert placer_solution.epoch_id("add") - placer_solution.epoch_id("exp") == 1

def test_override_chip_id(test_device):
    def matmul_buffer_matmul(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Buffer(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2

    # interactive_placer multi-chip is only enabled for B0
    if not test_device.is_wormhole_b0():
        pytest.skip()

    tt0 = TTDevice("tt0", devtype=BackendType.Golden, chip_ids=[1, 2, 3, 0])
    shape = (1, 1, 64, 64)
    act1 = Tensor.create_from_torch(torch.rand(shape, requires_grad=True)) 
    module = ModuleBuilder(
        matmul_buffer_matmul,
        ff1_weights=pybuda.Tensor.create_from_torch(torch.rand(shape)),
        ff2_weights=pybuda.Tensor.create_from_torch(torch.rand(shape))
    )
    tt0.place_module(module)

    # apply overrides
    pybuda.config.override_op_placement("ff1", chip_id=3)
    pybuda.config.override_op_placement("gelu", chip_id=1)
    pybuda.config.override_op_placement("ff2", chip_id=2)

    compile_result = pybuda_compile(tt0, "test_override_chip_id", act1)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    assert placer_solution.name_to_op_placement["ff1"].chip_id == 3
    assert placer_solution.name_to_op_placement["gelu"].chip_id == 1
    assert placer_solution.name_to_op_placement["ff2"].chip_id == 2
    assert placer_solution.name_to_op_placement["ff2_output_nop_0"].chip_id == 0

    assert len(placer_solution.epoch_id_to_chip) == 4

def test_epoch_break_fork_join():
    training = False

    mod = ForkJoinTest("fork_join_test")
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))
    act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))
    mod.set_parameter("weights2", torch.rand(*BudaTest.shape, requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    compiler_cfg = CompilerConfig(enable_training=training, compile_depth=CompileDepth.BALANCER_PASS)
    # currently the schedule yields:
    # ["nop", "matmul2", "matmul1","matmul2_nop", "matmul1_nop", "add"]
    # so we should expect ["nop", "matmul2"], ["matmul2_nop", "matmul1", "matmul1_nop", "add"]
    compiler_cfg.place_on_new_epoch(["matmul1_nop", "matmul2_nop"])

    compile_result = pybuda_compile(tt0, "sanity", act1, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    #for name, op_placement in placer_solution.name_to_op_placement.items():
    #    print(f"{name}: op_placement: {op_placement}")

    assert placer_solution.epoch_id("nop") == 0
    assert placer_solution.epoch_id("matmul2") == 0
    assert placer_solution.epoch_id("matmul1") == 0
    assert placer_solution.epoch_id("matmul1_nop") == 1
    assert placer_solution.epoch_id("matmul2_nop") == 1
    assert placer_solution.epoch_id("add") == 1

def test_change_start_grid_location(test_kind):
    def matmul_buffer_matmul(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Buffer(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2
    
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    shape = (1, 1, 64, 64)
    act1 = Tensor.create_from_torch(torch.rand(shape, requires_grad=True)) 
    module = ModuleBuilder(
        matmul_buffer_matmul,
        ff1_weights=pybuda.Tensor.create_from_torch(torch.rand(shape)),
        ff2_weights=pybuda.Tensor.create_from_torch(torch.rand(shape))
    )
    tt0.place_module(module)

    # apply overrides
    pybuda.config.override_op_size("ff2", (2,2))   
    pybuda.config.override_op_placement("ff2", start=(3,3))

    compile_result = pybuda_compile(tt0, "test_change_start_grid_location", act1)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]
    placed_core = placer_solution.name_to_op_placement["ff2"].placed_cores

    assert placed_core.start.row == 3
    assert placed_core.start.col == 3


def test_conflicting_placement_overrides(test_kind):
    def conflicting_placement_overrides(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Buffer(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2
    
    pybuda.config.override_op_placement("gelu", start=[2,2])
    pybuda.config.override_op_placement("ff2", start=[2,2])

    module = ModuleBuilder(conflicting_placement_overrides, ff1_weights=pybuda.Parameter(1,1,64,64), ff2_weights=pybuda.Parameter(1,1,64,64))
    verify.verify_module(module, [(1, 1, 64, 64)], VerifyConfig(test_kind=test_kind))


def test_dram_allocator_api(test_device):
    shape = (1, 1, 32, 32)
    test_kind = verify.TestKind.INFERENCE

    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def override_dram_allocator(x, weight=None):
        mm0 = pybuda.op.Matmul("mm0", x, weight)
        gelu = pybuda.op.Gelu("gelu", mm0)
        return gelu

    x = Tensor.create_from_torch(torch.randn(shape))
    w = pybuda.Parameter(torch.randn(shape))
    pybuda.config.set_epoch_break("gelu")
    pybuda.config.override_dram_queue_placement("e2e_mm0_0", chip_id=0)
    override_dram_allocator(x, weight=w)


@pytest.mark.parametrize("transpose_op", [False, True])
@pytest.mark.parametrize("temporal_epoch_break", [False, True])
def test_predicate_overrides(test_device, transpose_op, temporal_epoch_break):
    shape = (1, 1, 64, 64)
    num_layers = 3
    chip_ids = [0]

    pybuda.config.override_op_placement(
        query.name_regex("mm\\d"),
        transpose_op=transpose_op,
        temporal_epoch_break=temporal_epoch_break,
    )

    for layer in range(num_layers):
        pybuda.config.override_op_size(f"mm{layer}", (1, 2))

    @compile(chip_ids=chip_ids)
    def predicate_override(x, weight=None):
        out = x
        for layer in range(num_layers):
            out = pybuda.op.Matmul(f"mm{layer}", out, weight)
            out = pybuda.op.Exp(f"exp{layer}", out)
        return out

    x = Tensor.create_from_torch(torch.randn(shape))
    w = pybuda.Parameter(torch.randn(shape))
    compile_result = predicate_override(x, weight=w)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    mm0 = placer_solution.name_to_op_placement["mm0"]
    mm1 = placer_solution.name_to_op_placement["mm1"]
    mm2 = placer_solution.name_to_op_placement["mm2"]
    if transpose_op:
        assert mm0.grid_transpose
        assert mm1.grid_transpose
        assert mm2.grid_transpose
    if temporal_epoch_break:
        assert mm1.epoch_id > mm0.epoch_id
        assert mm2.epoch_id > mm1.epoch_id
