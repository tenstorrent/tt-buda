# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import os 
import math

import pybuda
import pybuda.op
import pybuda.op.nn as nn

from pybuda import PyBudaModule, PyTorchModule, TTDevice, Tensor, Parameter, pybuda_compile, CompilerConfig, SGD 
from pybuda.verify import verify_module, VerifyConfig, TestKind
from pybuda.ttdevice import get_device_config
from pybuda.config import _get_global_compiler_config
from pybuda._C.backend_api import BackendType, BackendDevice
from transformers import BertModel, BertConfig

verify_cfg = VerifyConfig(run_golden=True) # Run backend golden check on all tests in here

def get_relaxed_atol_pcc_bert_encoder(test_kind, test_device, microbatch_size = 1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.95
    if test_device.is_silicon():
        training_pcc = 0.85
        if microbatch_size > 1:
            training_atol = 0.55
            training_pcc = 0.8
    inference_atol = 0.1
    inference_pcc = 0.95
    relative_atol = training_atol if test_kind.is_training() else inference_atol
    if test_device.is_silicon() and test_kind.is_training():
        relative_atol *= 3.5
    pcc = training_pcc if test_kind.is_training() else inference_pcc

    return relative_atol, pcc

  
class ExpModule(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
            
    def forward(self, act1):  
        return pybuda.op.Exp("exp", act1)

 
class TwoOpsModule(pybuda.PyBudaModule):
    def __init__(self, name, r, c):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(torch.rand(r*32, c*32), requires_grad=True)
            
    def forward(self, act1, act2):
        e1 = pybuda.op.Exp("exp", act1)
        m1 = pybuda.op.Matmul("matmul", e1, self.weights1)  
        t1 = pybuda.op.Transpose("transpose", m1, 1,2)
        a1 = pybuda.op.Add("add", t1, act2)
        return a1
 
class ThreeOpsModule(pybuda.PyBudaModule):
    def __init__(self, name, r, c):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(torch.rand(r*32, c*32), requires_grad=True)
            
    def forward(self, act1, act2, act3):
        e1 = pybuda.op.Exp("exp1", act1) 
        a1 = pybuda.op.Add("add1", e1, act2) 
        e2 = pybuda.op.Exp("exp2", act3) 
        c1 = pybuda.op.Concatenate("concat", a1,a1, axis=1) 
        m1 = pybuda.op.Matmul("matmul", e2, self.weights1)   
        a2 = pybuda.op.Add("add2", c1, m1)        
        return a2

class FourOpsModule(pybuda.PyBudaModule):
    def __init__(self, name, r, c):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(torch.rand(r*32, c*32), requires_grad=True)
            
    def forward(self, act1, act2, act3):
        e1 = pybuda.op.Exp("exp1", act1) 
        e2 = pybuda.op.Exp("exp2", act2) 
        a1 = pybuda.op.Add("add1", e1, e1)  
        c1 = pybuda.op.Concatenate("concat", a1, e2, axis=1) 
        m1 = pybuda.op.Matmul("matmul", c1, self.weights1)  
        a2 = pybuda.op.Add("add2", act3, m1)        
        return a2
  
class TwoOpsNoTModule(pybuda.PyBudaModule):
    def __init__(self, name, r, c):
        super().__init__(name) 
        self.weights1 = pybuda.Parameter(torch.rand(r*32, c*32), requires_grad=True)
   
    def forward(self, act1, act2):
        e1 = pybuda.op.Exp("exp", act1) 
        m1 = pybuda.op.Matmul("matmul", e1, self.weights1)
        a1 = pybuda.op.Add("add", m1, act2)
        return a1

class TwoOpsModulev2(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name) 
            
    def forward(self, act1, act2):
        e1 = pybuda.op.Exp("exp1", act1)
        e2 = pybuda.op.Exp("exp2", act2)
        m1 = pybuda.op.Matmul("matmul", e1, e2)   
        return m1
 
@pytest.mark.parametrize("r", [x+1 for x in range(10)])
@pytest.mark.parametrize("c", [x+1 for x in range(10)])
def test_manual_op_transpose(test_device, r, c):
    if (test_device.arch == pybuda.BackendDevice.Wormhole_B0 or test_device.arch == pybuda.BackendDevice.Blackhole) and (r > 8 or c > 8):
        pytest.skip(f"{test_device.arch.to_string()} has 8 columns, skip the op-test with c = 9 or 10")

    compiler_cfg = _get_global_compiler_config()
    dev_cfg = get_device_config(test_device.arch, [0], compiler_cfg.backend_cluster_descriptor_path, compiler_cfg.backend_runtime_params_path, compiler_cfg.store_backend_db_to_yaml, test_device.devtype)
    if r == c or c > dev_cfg.grid_size.r or r > dev_cfg.grid_size.c:
        pytest.skip("op's r and c are the same, or invalid op-shape considering grid-size")
 
    pybuda.config.override_op_size("exp", (r, c))
    pybuda.config.override_op_placement("exp", transpose_op=True)

    mod = ExpModule("test_manual_T_module")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=test_device.arch)
    tt0.place_module(mod) 
 
    act1 = Tensor.create_from_torch(torch.rand((1, 1, r*32, c*32), requires_grad=True)) 

    compile_result = pybuda_compile(tt0, "sanity-manual_T", act1, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    placed_core = placer_solution.name_to_op_placement["exp"].placed_cores
    grid_transpose = placer_solution.name_to_op_placement["exp"].grid_transpose
   
    assert placed_core.size_r() == c and placed_core.size_c() == r
    assert grid_transpose == True   


def test_auto_op_transpose_case1(test_device): 
    compiler_cfg = _get_global_compiler_config()
 
    pybuda.config.override_op_size("exp", (1, 3))   
    pybuda.config.override_op_size("add", (2, 1)) 
    pybuda.set_configuration_options(enable_auto_transposing_placement=True)

    mod = TwoOpsModule("test_auto_T1", 3, 2)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=test_device.arch)
    tt0.place_module(mod) 
 
    act1 = Tensor.create_from_torch(torch.rand((1, 32, 96), requires_grad=True)) 
    act2 = Tensor.create_from_torch(torch.rand((1, 64, 32), requires_grad=True))    
   
    compile_result = pybuda_compile(tt0, "sanity-auto_T1", act1, act2, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    placed_core = placer_solution.name_to_op_placement["add"].placed_cores
    grid_transpose = placer_solution.name_to_op_placement["add"].grid_transpose

    expected_placed_core_rows = (0,1)
    # if grayskull, there is 1 more op, since transpose combined with srcA isn't supported
    expected_placed_core_cols = (5,7) if test_device.is_grayskull() else (4,6)

    assert (placed_core.start.row, placed_core.end.row) == expected_placed_core_rows, f"(placed_core.start.row, placed_core.end.row) = {(placed_core.start.row, placed_core.end.row)} != expected_placed_core_rows = ({expected_placed_core_rows})"
    assert (placed_core.start.col, placed_core.end.col) == expected_placed_core_cols, f"(placed_core.start.col, placed_core.end.col) = {(placed_core.start.col, placed_core.end.col)} != expected_placed_core_cols = ({expected_placed_core_cols})"
    assert placed_core.size_r() == 1 and placed_core.size_c() == 2, f"placed_core.start.row = {placed_core.start.row} != 0 or placed_core.start.col = {placed_core.start.col} != 5"
    assert grid_transpose == True   


def test_auto_op_transpose_case2(test_device):
    compiler_cfg = _get_global_compiler_config()
 
    pybuda.config.override_op_size("exp", (10, 1))   
    pybuda.config.override_op_size("add", (10, 1)) 
    pybuda.set_configuration_options(enable_auto_transposing_placement=True)

    mod = TwoOpsNoTModule("test_auto_T2", 1, 1)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=test_device.arch)
    tt0.place_module(mod) 
 
    act1 = Tensor.create_from_torch(torch.rand((1, 10*32, 32), requires_grad=True)) 
    act2 = Tensor.create_from_torch(torch.rand((1, 10*32, 32), requires_grad=True))    
   
    compile_result = pybuda_compile(tt0, "sanity-auto_T2", act1, act2, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    placed_core = placer_solution.name_to_op_placement["add"].placed_cores
    grid_transpose = placer_solution.name_to_op_placement["add"].grid_transpose

    if test_device.arch == BackendDevice.Grayskull:
        assert placed_core.size_r() == 1 and placed_core.size_c() == 10
        assert grid_transpose == True
    else:
        assert placed_core.size_r() == 10 and placed_core.size_c() == 1
        assert grid_transpose == False


def test_auto_op_transpose_case3(test_device): 
    compiler_cfg = _get_global_compiler_config()  
 
    pybuda.config.override_op_size("exp1", (2, 7))   
    pybuda.config.override_op_size("add1", (2, 7)) 
    pybuda.config.override_op_size("exp2", (4, 1))
    pybuda.set_configuration_options(enable_auto_transposing_placement=True)

    mod = ThreeOpsModule("test_auto_T3", 1, 7)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=test_device.arch)
    tt0.place_module(mod) 
 
    act1 = Tensor.create_from_torch(torch.rand((1, 2*32, 7*32), requires_grad=True)) 
    act2 = Tensor.create_from_torch(torch.rand((1, 2*32, 7*32), requires_grad=True))    
    act3 = Tensor.create_from_torch(torch.rand((1, 4*32, 32), requires_grad=True))

    compile_result = pybuda_compile(tt0, "sanity-auto_T3", act1, act2, act3, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    placed_core = placer_solution.name_to_op_placement["exp2"].placed_cores
    grid_transpose = placer_solution.name_to_op_placement["exp2"].grid_transpose
   
    assert placed_core.start.row == 0 and placed_core.start.col == 7
    assert placed_core.end.row == 4 and placed_core.end.col == 8  
    assert placed_core.size_r() == 4 and placed_core.size_c() == 1
    assert grid_transpose == False


def test_auto_op_transpose_multi_rows1(test_device): 
    if test_device.arch != pybuda.BackendDevice.Grayskull:
        pytest.skip("Targetting grid-size of GS only")

    compiler_cfg = _get_global_compiler_config()  
 
    pybuda.config.override_op_size("exp1", (4, 5))   
    pybuda.config.override_op_size("exp2", (1, 5))
    pybuda.config.override_op_size("add1", (4, 5)) 
    pybuda.config.override_op_size("add2", (5, 2))
    pybuda.set_configuration_options(enable_auto_transposing_placement=True)

    mod = FourOpsModule("test_auto_T4", 5, 2)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=test_device.arch)
    tt0.place_module(mod) 
 
    act1 = Tensor.create_from_torch(torch.rand((1, 4*32, 5*32), requires_grad=True)) 
    act2 = Tensor.create_from_torch(torch.rand((1, 1*32, 5*32), requires_grad=True))    
    act3 = Tensor.create_from_torch(torch.rand((1, 5*32, 2*32), requires_grad=True))

    compile_result = pybuda_compile(tt0, "sanity-auto_T4", act1, act2, act3, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    placed_core = placer_solution.name_to_op_placement["add2"].placed_cores
    grid_transpose = placer_solution.name_to_op_placement["add2"].grid_transpose
   
    assert placed_core.start.row == 1 and placed_core.start.col == 10
    assert placed_core.end.row == 6 and placed_core.end.col == 12  
    assert placed_core.size_r() == 5 and placed_core.size_c() == 2
    assert grid_transpose == False


def test_auto_op_transpose_multi_rows2(test_device): 
    if test_device.arch != pybuda.BackendDevice.Grayskull:
        pytest.skip("Targetting grid-size of GS only")

    compiler_cfg = _get_global_compiler_config()  
 
    pybuda.config.override_op_size("exp1", (3, 8))   
    pybuda.config.override_op_size("exp2", (8, 1))
    pybuda.set_configuration_options(enable_auto_transposing_placement=True)

    mod = TwoOpsModulev2("test_auto_T5")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=test_device.arch)
    tt0.place_module(mod) 
 
    act1 = Tensor.create_from_torch(torch.rand((1, 3*32, 8*32), requires_grad=True)) 
    act2 = Tensor.create_from_torch(torch.rand((1, 8*32, 1*32), requires_grad=True))     

    compile_result = pybuda_compile(tt0, "sanity-auto_T5", act1, act2, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg)
    placer_solution = compile_result.pass_specific_output_kwargs["placer_solution"]

    placed_core = placer_solution.name_to_op_placement["exp2"].placed_cores
    grid_transpose = placer_solution.name_to_op_placement["exp2"].grid_transpose
   
    assert placed_core.start.row == 0 and placed_core.start.col == 8
    assert placed_core.end.row == 8 and placed_core.end.col == 9  
    assert placed_core.size_r() == 8 and placed_core.size_c() == 1
    assert grid_transpose == False
 
