# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of matmul operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (+)  2.1 From another op
#       - Operator -> input
# (+)  2.2 From tm edge
#       - Combination: operator -> tm -> input
#       - tm -> input
# (+)  2.3 From DRAM queue
#       - input_queue flag = false
#       - Special case of From host? May it be triggered if the operator is not the first node of the network?
#       - Can this be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# (+)  2.4 From DRAM, but prologued (constant)
#       - Constants must be small enough to fit into L1
#       - Verification via netlists that scenario is triggered
#       - Input are not prologued for microbatch size = 1
# (+)  2.5 Const Inputs (const eval pass)
#       - Operator where all inputs are constants. Does it make difference if tensor is big > L1
#       - Verification via netlists that scenario is triggered???
# (+)  2.6 From host
#       - Input tensor as input of network -> Operator is first node in network and input_queue flag = true
#       - Can this scenario be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# 3 Operand shapes type(s):
# (+)  3.1 Full tensor (i.e. full expected shape)
#       - Is 3 dims max for all ops? Ex. Conv is 3d max
# (+)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (/)  3.3 Scalar
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (+)  4.1 Divisible by 32
# (+)  4.2 Prime numbers
# (+)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (+)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (/)  5.1 Output DF
# (/)  5.2 Intermediate DF
# (/)  5.3 Accumulation DF
# (+)  5.4 Operand DFs
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example


import os
from typing import Dict, List
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from pybuda.verify.config import TestKind

from pybuda.verify.backend import verify_module

from test.common import ModuleBuilder

from test.common import run

from pybuda.tensor import Tensor
import torch

from pybuda.module import PyBudaModule

from pybuda.config import _get_global_compiler_config

from test.test_sanity import get_device_intermediates

from pybuda.op.eval.common import compare_tensor_to_golden

from test.operators.utils import netlist_utils
from test.operators.utils import FailingReasons

from .models import generic
from .models import custom
from .models import special_cases
from .models import test_plan



MODELS_PATH = "./pybuda/test/operators/matmul/models/"
MODELS_GENERIC_PATH = MODELS_PATH + "generic/" 
MODELS_CUSTOM_PATH = MODELS_PATH + "custom/"
MODELS_SPECIAL_CASES_PATH = MODELS_PATH + "special_cases/"
MODELS_TEST_PLAN_PATH = MODELS_PATH + "test_plan/"


SHAPE_NO = 1
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 7
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True

np.random.seed(1)

# tensors of arbitrarily shape
# shape = [np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, np.random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)).tolist() for _ in range(SHAPE_NO)]
shape = []
# 4-dimensional tensors
if SHAPE_FIXED:
    size = 4
    shape_fixed = []
    for _ in range(SHAPE_NO):
        wdim = 1 if WDIM_FIXED else np.random.randint(SHAPE_WDIM_MIN, SHAPE_WDIM_MAX)
        zdim = np.random.randint(SHAPE_ZDIM_MIN, SHAPE_ZDIM_MAX)
        sh = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, 2).tolist()
        shape_fixed.append([wdim, zdim] + sh)
    shape += shape_fixed

# Generic Shape

#@pytest.mark.xfail(
#    reason="tenstorrent/pybuda#22"
#)
@pytest.mark.parametrize("shape", shape, ids=[f"shape{'x'.join([str(jtem) for jtem in item])}" for item in shape])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_GENERIC_PATH) if "model" in item])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def test_matmul_generic(
    op_test_kind,
    model,
    shape
):
    test_kind = op_test_kind
    if test_kind.is_training() and len(shape) >= 3 and shape[-3] > 1:
        pytest.skip("Matmul with gradient accumulate must have t=1")

    architecture = f'generic.{model}.BudaMatmulTest(shape={shape})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    print(model.get_parameters())
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute(),
                     ), 
        verify_cfg=VerifyConfig()
    )


# Custom Shape

@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_CUSTOM_PATH) if "model" in item])
def test_matmul_custom(
    test_kind,
    model
):
    if test_kind.is_training():
        pytest.xfail() # numbers gets too big

    if model == "model_4":
        pytest.xfail() # balancer failure

    architecture = f'custom.{model}.BudaMatmulTest()'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute(),
                     ), 
        verify_cfg=VerifyConfig()
    )

# test matmul op in cases where input tensors are as described in pytorch docs
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_SPECIAL_CASES_PATH) if "model" in item])
def test_matmul_according_to_pytorch_docs(
    model,
    test_device
):

    # TODO Unify models 11 to 15 by parametrizing the input shapes

    #BUG
    if model in ("model_11", ):
        # Matmul op when two input tensors are vectors is not supported. Error: pybuda/pybuda/op/eval/pybuda/matmul.py:135: E    IndexError: list index out of range
        pytest.xfail(reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE)
    #BUG
    if model in ("model_12", ):
        # Matmul op when two input tensors are matrix(without microbatch size) is not supported. Error: pybuda/pybuda/op/eval/pybuda/matmul.py:29: E     RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 1x7)
        pytest.xfail(reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE)
    #BUG
    if model in ("model_13", ):
        # Matmul op if the first argument is 1-dimensional and the second argument is 2-dimensional is not supported. Error: pybuda/pybuda/tensor.py:383: E    AssertionError: Setting a tensor value of incorrect shape: (1, 7) vs torch.Size([7])
        pytest.xfail(reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE)
    #BUG
    if model in ("model_14", ):
        # Matmul op if the first argument is 2-dimensional and the second argument is 1-dimensional is not suppported. Error: pybuda/pybuda/op/eval/pybuda/matmul.py:29: E    RuntimeError: size mismatch, got input (1), mat (1x64), vec (1)
        pytest.xfail(reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE)
    #BUG
    if model in ("model_15", ):
        # Matmul op when one of the arguments is 1-dimensional and the other one is N-dimensional is not suppported. Error: pybuda/pybuda/op/eval/pybuda/matmul.py:29: E    RuntimeError: size mismatch, got input (32), mat (32x64), vec (1)
        pytest.xfail(reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE)

    architecture = f'special_cases.{model}.BudaMatmulTest()'
    model = eval(architecture)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_training = False
    compiler_cfg.input_queues_on_host = True

    verify_module(
        model,
        input_shapes=model.shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )



def get_input_shapes(microbatch_size1=1, microbatch_size2=1):
                                              # Here we cover interesting combinations of input shapes:
    return [
            (microbatch_size1, microbatch_size2, 3, 4),         # 3.1 Full tensor (i.e. full expected shape)
            (microbatch_size1, microbatch_size2, 45, 17),        # 3.1 Full tensor (i.e. full expected shape)
            (microbatch_size1, microbatch_size2, 1, 23),        # 3.2 Tensor reduce on one or more dims to 1
            (microbatch_size1, microbatch_size2, 64, 1),        # 3.2 Tensor reduce on one or more dims to 1
            (microbatch_size1, microbatch_size2, 100, 100),     # 4.3 Very large (thousands, 10s of thousands)
            (microbatch_size1, microbatch_size2, 1000, 100),    # 4.3 Very large (thousands, 10s of thousands)
            (microbatch_size1, microbatch_size2, 10, 1000),     # 4.4 Extreme ratios between height/width
            (microbatch_size1, microbatch_size2, 9920, 1),     # 4.4 Extreme ratios between height/width 
            (microbatch_size1, microbatch_size2, 10000, 1),     # 4.4 Extreme ratios between height/width   
            (microbatch_size1, microbatch_size2, 32, 64),       # 4.1 Divisible by 32
            (microbatch_size1, microbatch_size2, 160, 96),       # 4.1 Divisible by 32
            (microbatch_size1, microbatch_size2, 17, 41),       # 4.2 Prime numbers
            (microbatch_size1, microbatch_size2, 89, 3),       # 4.2 Prime numbers

            (microbatch_size1, 3, 4),         # 3.1 Full tensor (i.e. full expected shape)
            (microbatch_size1, 45, 17),        # 3.1 Full tensor (i.e. full expected shape)
            (microbatch_size1, 1, 23),        # 3.2 Tensor reduce on one or more dims to 1
            (microbatch_size1, 64, 1),        # 3.2 Tensor reduce on one or more dims to 1
            (microbatch_size1, 100, 100),     # 4.3 Very large (thousands, 10s of thousands)
            (microbatch_size1, 1000, 100),    # 4.3 Very large (thousands, 10s of thousands)
            (microbatch_size1, 10, 1000),     # 4.4 Extreme ratios between height/width
            (microbatch_size1, 9920, 1),     # 4.4 Extreme ratios between height/width  
            (microbatch_size1, 10000, 1),     # 4.4 Extreme ratios between height/width 
            (microbatch_size1, 32, 64),       # 4.1 Divisible by 32
            (microbatch_size1, 160, 96),       # 4.1 Divisible by 32
            (microbatch_size1, 17, 41),       # 4.2 Prime numbers
            (microbatch_size1, 89, 3),       # 4.2 Prime numbers
            ]

# test matmul in all cases according to test plan
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_TEST_PLAN_PATH) if "model" in item])
@pytest.mark.parametrize("input_shape", get_input_shapes())
def test_matmul_according_to_test_plan(
    model,
    input_shape,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    if(model == "model_op_src_const_inputs2" and math_fidelity == None):
        pytest.skip() # this model has its own test: test_matmul_dram_prologued

    #BUG: when input shape is (1, 1, 10000, 1) - extreme ratios between height/width; it works for input shape when one dimension is 9920 or less, everything above(like 10000) throws error
    if (input_shape == (1, 1, 10000, 1) or input_shape == (1, 10000, 1)) and model in (
            "model_op_src_from_another_op",
            "model_op_src_from_dram2",
            "model_op_src_const_inputs1",
            "model_op_src_const_inputs2",
            "model_op_src_from_host",
        ):
        # Error for input shape (1, 1, 10000, 1). Error message: RuntimeError: TT_ASSERT @ pybuda/csrc/placer/lower_to_placer.cpp:245:
        pytest.xfail(reason=FailingReasons.COMPILATION_FAILED)

    # generate input shapes for every model
    opernad_num = 0
    tr_operand_num = 0
    match model:
        case "model_op_src_from_another_op":
            opernad_num = 2
            tr_operand_num = 2
        case "model_op_src_from_tm_edge1":
            opernad_num = 1
            tr_operand_num = 1
        case "model_op_src_from_tm_edge2":
            opernad_num = 2
            tr_operand_num = 2
        case "model_op_src_from_dram1":
            opernad_num = 0
            tr_operand_num = 1
        case "model_op_src_from_dram2":
            opernad_num = 1
            tr_operand_num = 1
        case "model_op_src_const_inputs1":
            opernad_num = 1
            tr_operand_num = 1
        case "model_op_src_const_inputs2":
            opernad_num = 0
            tr_operand_num = 1
        case "model_op_src_from_host": 
            opernad_num = 1
            tr_operand_num = 1
        case _:
            pytest.skip(f'Unsupported model for matmul op test: {model}')
    if(len(input_shape) == 3):
        tr = (input_shape[0],input_shape[2],input_shape[1])
    else:
        tr = (input_shape[0],input_shape[1],input_shape[3],input_shape[2])
    input_shapes = list([input_shape for _ in range(opernad_num)])
    for _ in range(tr_operand_num):
        input_shapes.append(tr) 
    input_shapes = tuple(input_shapes)


    match model:
        case "model_op_src_from_dram1":
            input_shape = (1,) + input_shape[1:]
            architecture = f'test_plan.{model}.BudaMatmulTest({input_shape})'
        case "model_op_src_const_inputs1": 
            input_shape = (1,) + input_shape[1:]
            tr = (1,) + tr[1:]
            architecture = f'test_plan.{model}.BudaMatmulTest({input_shape}, {tr})'
        case "model_op_src_const_inputs2":
            input_shape = (1,) + input_shape[1:]
            architecture = f'test_plan.{model}.BudaMatmulTest({input_shape})'
        case _:
            architecture = f'test_plan.{model}.BudaMatmulTest()'
    model_eval = eval(architecture)


    # set compiler config file based on model we are testing
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_training = False
    match model:
        case "model_op_src_from_dram2":
            compiler_cfg.input_queues_on_host = False
        case _:
            compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    
    verify_module(
        model_eval,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )

    file_path = pybuda.pybudaglobal.get_devices()[0]._compile_output.netlist_filename
    match model:
        case "model_op_src_from_dram2":
            assert netlist_utils.read_netlist_value(file_path, "/queues/x1/loc") == 'dram'
            assert netlist_utils.read_netlist_value(file_path, "/queues/x2/loc") == 'dram'
        case "model_op_src_const_inputs1":
            d = netlist_utils.read_netlist_value(file_path, "/graphs/fwd_0_0_temporal_epoch_0")
            for key in d.keys():
                assert "Matmul" not in key



def get_input_shapes_prologued():
                                              # Here we cover interesting combinations of input shapes:
    return [
            ((2, 3, 4),         True, False),  #0        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 3, 4),         False, True),  #1        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 3, 4),         None, True),   #2        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 3, 4),         True, False),  #3        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 3, 4),         False, True),  #4        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 3, 4),         None, True),   #5        # 3.1 Full tensor (i.e. full expected shape) ! not working as described in docs
            ((2, 45, 17),       None, True),   #6        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 1, 23),        None, True),   #7        # 3.2 Tensor reduce on one or more dims to 1
            ((2, 64, 1),        None, True),   #8        # 3.2 Tensor reduce on one or more dims to 1
            ((2, 100, 100),     None, True),   #9        # 4.3 Very large (thousands, 10s of thousands)
            ((2, 1000, 100),    None, True),   #10       # 4.3 Very large (thousands, 10s of thousands)
            ((2, 10, 1000),     None, True),   #11       # 4.4 Extreme ratios between height/width
            ((2, 9920, 1),      None, True),   #12       # 4.4 Extreme ratios between height/width
            ((2, 10000, 1),     None, False),  #13       # 4.4 Extreme ratios between height/width
            ((2, 32, 64),       None, True),   #14       # 4.1 Divisible by 32
            ((2, 160, 96),      None, True),   #15       # 4.1 Divisible by 32
            ((2, 17, 41),       None, True),   #16       # 4.2 Prime numbers
            ((2, 89, 3),        None, True),   #17       # 4.2 Prime numbers

            ((2, 1, 3, 4),      True, False),  #18       # 3.1 Full tensor (i.e. full expected shape)
            ((2, 1, 3, 4),      False, True),  #19       # 3.1 Full tensor (i.e. full expected shape)
            ((2, 1, 3, 4),      None, True) ,  #20       # 3.1 Full tensor (i.e. full expected shape)
            ((1, 1, 3, 4),      True, False),  #21       # 3.1 Full tensor (i.e. full expected shape)
            ((1, 1, 3, 4),      False, True),  #22       # 3.1 Full tensor (i.e. full expected shape)
            ((1, 1, 3, 4),      None, True),   #23       # 3.1 Full tensor (i.e. full expected shape) ! not working as described in docs
            ((2, 1, 45, 17),    None, True) ,  #24       # 3.1 Full tensor (i.e. full expected shape)
            ((2, 1, 1, 23),     None, True) ,  #25       # 3.2 Tensor reduce on one or more dims to 1
            ((2, 1, 64, 1),     None, True) ,  #26       # 3.2 Tensor reduce on one or more dims to 1
            ((2, 1, 100, 100),  None, True) ,  #27       # 4.3 Very large (thousands, 10s of thousands)
            ((2, 1, 1000, 100), None, True) ,  #28       # 4.3 Very large (thousands, 10s of thousands)
            ((2, 1, 10, 1000),  None, True) ,  #29       # 4.4 Extreme ratios between height/width
            ((2, 1, 9920, 1),   None, True) ,  #30       # 4.4 Extreme ratios between height/width 
            ((2, 1, 10000, 1),  None, True) ,  #31       # 4.4 Extreme ratios between height/width   
            ((2, 1, 32, 64),    None, True) ,  #32       # 4.1 Divisible by 32
            ((2, 1, 160, 96),   None, True) ,  #33       # 4.1 Divisible by 32
            ((2, 1, 17, 41),    None, True) ,  #34       # 4.2 Prime numbers
            ((2, 1, 89, 3),     None, True) ,  #35       # 4.2 Prime numbers
            ]

@pytest.mark.parametrize("input_shape, default_dram_params, prologue", get_input_shapes_prologued())
def test_matmul_dram_prologued(
    input_shape,
    default_dram_params,
    prologue,
    test_device,
):
    model = "model_op_src_const_inputs2"
    #BUG: when input shape is (2, 1, 10000, 1) or (2, 10000, 1) - extreme ratios between height/width; it works for input shape when one dimension is 9920 or less, everything above(like 10000) throws error
    if (input_shape == (2, 1, 10000, 1) or input_shape == (2, 10000, 1)) and model == "model_op_src_const_inputs2":
        # Error for input shape (1, 1, 10000, 1). Error message: RuntimeError: TT_ASSERT @ pybuda/csrc/placer/lower_to_placer.cpp:245:
        pytest.xfail(reason=FailingReasons.COMPILATION_FAILED)
   
    # generate input shapes
    opernad_num = 0
    tr_operand_num = 1
    if(len(input_shape) == 3):
        tr = (input_shape[0],input_shape[2],input_shape[1])
    else:
        tr = (input_shape[0],input_shape[1],input_shape[3],input_shape[2])
    input_shapes = list([input_shape for _ in range(opernad_num)])
    for _ in range(tr_operand_num):
        input_shapes.append(tr) 
    input_shapes = tuple(input_shapes)

    input_shape = (1,) + input_shape[1:]

    architecture = f'test_plan.{model}.BudaMatmulTest({input_shape})'
    model_eval = eval(architecture)

    # set compiler config file
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_training = False
    compiler_cfg.input_queues_on_host = False
    compiler_cfg.default_dram_parameters = default_dram_params
    
    verify_module(
        model_eval,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )

    file_path = pybuda.pybudaglobal.get_devices()[0]._compile_output.netlist_filename
    d = netlist_utils.read_netlist_value(file_path, "/programs/0/run_fwd_0/4/execute/queue_settings/input_0_mm1")
    if prologue:
        assert d['prologue']
    else:
        assert not d['prologue']


def get_input_shape(microbatch_size1=1, microbatch_size2=1):
    return (microbatch_size1, microbatch_size2, 11, 37)

verify_input_params=[ 
                        {"dev_data_format": pybuda.DataFormat.Float16_b},
                    ]

compiler_math_fidelity = [
                            pybuda.MathFidelity.LoFi,
                            pybuda.MathFidelity.HiFi2,
                            pybuda.MathFidelity.HiFi3,
                            pybuda.MathFidelity.HiFi4,
                         ]

@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_TEST_PLAN_PATH) if "model" in item])
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_matmul_mf_inputs(model, test_device, math_fidelity):
    test_matmul_according_to_test_plan(model, get_input_shape(), test_device, verify_input_params, math_fidelity);



verify_input_params=[
                        {"dev_data_format": pybuda.DataFormat.Bfp2},
                        {"dev_data_format": pybuda.DataFormat.Bfp2_b},
                        {"dev_data_format": pybuda.DataFormat.Bfp4},
                        {"dev_data_format": pybuda.DataFormat.Bfp4_b},
                        {"dev_data_format": pybuda.DataFormat.Bfp8},
                        {"dev_data_format": pybuda.DataFormat.Bfp8_b},
                        {"dev_data_format": pybuda.DataFormat.Float16},  
                        {"dev_data_format": pybuda.DataFormat.Float16_b},
                        {"dev_data_format": pybuda.DataFormat.Float32},
                        {"dev_data_format": pybuda.DataFormat.Int8},
                        {"dev_data_format": pybuda.DataFormat.Lf8},
                        {"dev_data_format": pybuda.DataFormat.RawUInt16},
                        {"dev_data_format": pybuda.DataFormat.RawUInt32},
                        {"dev_data_format": pybuda.DataFormat.RawUInt8},
                        {"dev_data_format": pybuda.DataFormat.UInt16},
                    ]
compiler_math_fidelity = pybuda.MathFidelity.HiFi4

@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_TEST_PLAN_PATH) if "model" in item])
@pytest.mark.parametrize("input_params", verify_input_params)
def test_matmul_df_inputs(model, test_device, input_params):
    test_matmul_according_to_test_plan(model, get_input_shape(), test_device, input_params, compiler_math_fidelity);


# from sanity
def test_matmul_relu(
    test_kind
):
    if test_kind.is_training():
        pytest.xfail() # numbers gets too big

    def matmul_relu(act, *, weights):
        op0 = pybuda.op.Matmul(f"op0", act, weights)
        op1 = pybuda.op.Relu(f"op1", op0)
        return op1
    
    module = ModuleBuilder(matmul_relu, weights=pybuda.Parameter(1,1,64,64))
    
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=test_kind))
    
def test_matmul_gradient_t(test_kind, test_device):
    shape = (1, 3, 128, 128)

    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc),
    )
    def simple_matmul_gradient_t(x, weight=None):
        return pybuda.op.Matmul("mm0", x, weight)

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training()))
    w = pybuda.Parameter(*shape, requires_grad=test_kind.is_training())
    simple_matmul_gradient_t(x, weight=w)

def test_matmul_gelu_matmul(test_kind):
    def matmul_gelu(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Gelu(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2

    module = ModuleBuilder(matmul_gelu, ff1_weights=pybuda.Parameter(1,1,64,64), ff2_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=test_kind, optimizer=None))
    
def test_matmul_buffer_matmul(test_kind):
    def matmul_buffer_matmul(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Buffer(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2
    
    pybuda.set_epoch_break("gelu")
    pybuda.set_epoch_break("ff2")

    module = ModuleBuilder(matmul_buffer_matmul, ff1_weights=pybuda.Parameter(1,1,64,64), ff2_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)], VerifyConfig(test_kind=test_kind))


def test_multipliers_overrides(test_device):
    shape = (1, 1, 32, 32)
    test_kind = TestKind.INFERENCE

    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def simple_matmul_buffer_overrides(x, weight=None):
        return pybuda.op.Matmul("mm0", x, weight)

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training()))
    w = pybuda.Parameter(torch.randn(shape, requires_grad=test_kind.is_training()))
    pybuda.config.override_input_buffer_multiplier("mm0", 0, multiplier=4)
    pybuda.config.internal_override_output_buffer_multiplier("mm0", multiplier=4)

    simple_matmul_buffer_overrides(x, weight=w)

def test_scalar_matmul_bias(test_device):
    pybuda.set_configuration_options(backend_output_dir=f"tt_build/test_scalar_matmul_bias")
    @run(test_device)
    def scalar_matmul_bias(a, w=None, b=None):
        x = pybuda.op.Matmul("", a, w)
        x = pybuda.op.Add("", x, b)
        return x
    
    x = Tensor.create_from_torch(torch.randn(1, 1, 32, 32))
    w = pybuda.Parameter.create_from_torch(torch.randn(1, 1, 32, 128))
    tmp = torch.zeros(1, 1, 1, 1)
    tmp[0, 0, 0, 0] = 1000.0
    b = pybuda.Parameter.create_from_torch(tmp)
    scalar_matmul_bias(x, w=w, b=b)


def test_read_back_intermediates(test_kind, test_device):
    if test_kind.is_training():
        op_intermediates = ["matmul_intermediate", "bw_in0_matmul_output_matmul_1"]
    else:
        op_intermediates = ["matmul_intermediate"]

    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"]  = "1" #issue #2657
    pybuda.set_configuration_options(op_intermediates_to_save=op_intermediates)
    num_inputs = 4

    @run(
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            intermediates=True,
            microbatch_count=num_inputs,
        ),
        num_inputs=num_inputs,
    )
    def fetch_intermediates(x0, x1, x2):
        intermediate = pybuda.op.Matmul("matmul_intermediate", x0, x1)
        return pybuda.op.Matmul("matmul_output", intermediate, x2)

    x = Tensor.create_from_torch(torch.randn(1, 1, 63, 63, requires_grad=test_kind.is_training()))
    y = Tensor.create_from_torch(torch.randn(1, 1, 63, 63, requires_grad=test_kind.is_training()))
    z = Tensor.create_from_torch(torch.randn(1, 1, 63, 63, requires_grad=test_kind.is_training()))
    fetch_intermediates(x, y, z)

    device = pybuda.get_tenstorrent_device()
    compiled_results = device.get_compiled_results()

    golden_intermediates: Dict[str, torch.Tensor ] = compiled_results.golden_intermediates  # golden replicated
    device_intermediates: Dict[str, List[torch.Tensor]] = get_device_intermediates(op_intermediates)

    for op_name in op_intermediates:
        assert (len(device_intermediates[op_name]) == num_inputs), f"Expected {num_inputs} intermediate tensors for {op_name}"
        if op_name in golden_intermediates:
            for idx in range(num_inputs):
                compare_tensor_to_golden(
                    op_name,
                    golden_intermediates[op_name],
                    device_intermediates[op_name][idx],
                    is_buda=True,
                )
# end from sanity