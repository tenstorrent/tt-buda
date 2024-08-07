# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise unary operators
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
# (/)  2.4 From DRAM, but prologued (constant)
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

# 10379 passed, 190 skipped, 4270 xfailed, 18 xpassed, 3 warnings in 2797.58s (0:46:37)

import os
from typing import Dict, List
import pytest
import numpy as np

import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig
from pybuda.verify.config import TestKind

from test.operators.utils import netlist_utils, InputSourceFlags, VerifyUtils
from test.operators.utils import FailingReasons
from test.conftest import TestDevice

from pybuda.module import PyBudaModule

from pybuda.op_repo.datatypes import TensorShape

from . import models
from .models import test_plan




TEST_PLAN_MODELS_PATH = "./pybuda/test/operators/eltwise_unary/models/test_plan/"



########## HELPER METHOD

def verify(
        test_device: TestDevice,
        input_model: PyBudaModule, 
        input_operator: str, 
        input_shape: TensorShape, 
        kwargs:Dict = {}, 
        input_params: List[Dict] = [], 
        input_dev_data_format: pybuda.DataFormat = None, 
        input_math_fidelity: pybuda.MathFidelity = None, 
        pcc: float = 0.99
    ):
    '''Common verification function for all tests'''
    
    architecture = f'test_plan.{input_model}.BudaElementWiseUnaryTest(operator=pybuda.op.{input_operator}, opname="{input_operator}", shape={input_shape}'
    for k, v in kwargs.items():
        architecture = f'{architecture}, {k}={v}'
    architecture = f'{architecture})'
    model = eval(architecture)

    input_shapes = tuple([input_shape])
    
    input_source_flag = None
    if input_model == "model_op_src_from_dram":
        input_source_flag = InputSourceFlags.FROM_DRAM
    elif input_model == "model_op_src_from_host":
        input_source_flag = InputSourceFlags.FROM_HOST
        
    VerifyUtils.verify(
        model=model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        pcc=pcc,
        input_source_flag=input_source_flag,
        dev_data_format=input_dev_data_format,
        math_fidelity=input_math_fidelity,
    )

    file_path = VerifyUtils.get_netlist_filename() 
    match model:
        case "model_op_src_from_dram":
            assert netlist_utils.read_netlist_value(file_path, "/queues/x1/loc") == 'dram'
        case "model_op_src_const_inputs1":
            d = netlist_utils.read_netlist_value(file_path, "/graphs/fwd_0_0_temporal_epoch_0")
            for key in d.keys():
                assert input_operator not in key



########## ALL UNARY OPERATORS THAT ARE TESTED

def get_eltwise_unary_operators():
    return [
        "Abs",
        "LeakyRelu",
        "Exp",
        "Identity",
        "Reciprocal",
        "Sigmoid",
        "Sqrt",
        "Gelu",
        "Log",
        "Relu",
        "Buffer",
        "Tanh",
        "Sine",
        "Cosine",
        "Argmax",  
        # "Dropout",      # have their own test
        # "LogicalNot",   # have their own test
        # "Tilize",       # have their own test 
        # "Pow",          # have their own test
        # "Clip",         # have their own test
        # "CumSum",       # have their own test
    ]




########## ALL INPUT SHAPES USED FOR EACH OPERATOR IN TESTS

def get_input_shapes():
    return [
        # 2 dim with microbatch size = 1
            (1, 4),                         #0      # 3.1 Full tensor (i.e. full expected shape)
            (1, 17),                        #1      # 3.1 Full tensor (i.e. full expected shape)
            (1, 23),                        #2      # 3.2 Tensor reduce on one or more dims to 1
            (1, 1),                         #3      # 3.2 Tensor reduce on one or more dims to 1
            (1, 100),                       #4      # 4.3 Very large (thousands, 10s of thousands)
            (1, 500),                       #5      # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000),                      #6      # 4.4 Extreme ratios between height/width
            (1, 1920),                      #7      # 4.4 Extreme ratios between height/width  
            (1, 10000),                     #8      # 4.4 Extreme ratios between height/width 
            (1, 64),                        #9      # 4.1 Divisible by 32
            (1, 96),                        #10     # 4.1 Divisible by 32
            (1, 41),                        #11     # 4.2 Prime numbers
            (1, 3),                         #12     # 4.2 Prime numbers
            
        # # 2 dim with microbatch size > 1
            (3, 4),                         #13      # 3.1 Full tensor (i.e. full expected shape)              
            (45, 17),                       #14      # 3.1 Full tensor (i.e. full expected shape)
            (2, 23),                        #15      # 3.2 Tensor reduce on one or more dims to 1
            (64, 1),                        #16      # 3.2 Tensor reduce on one or more dims to 1
            (100, 100),                     #17      # 4.3 Very large (thousands, 10s of thousands)
            (1000, 100),                    #18      # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000),                     #19      # 4.4 Extreme ratios between height/width
            (9920, 1),                      #20      # 4.4 Extreme ratios between height/width  
            (10000, 1),                     #21      # 4.4 Extreme ratios between height/width 
            (32, 64),                       #22      # 4.1 Divisible by 32
            (160, 96),                      #23      # 4.1 Divisible by 32
            (17, 41),                       #24      # 4.2 Prime numbers
            (89, 3),                        #25      # 4.2 Prime numbers

        # 3 dim with microbatch size = 1
            (1, 3, 4),                      #26     # 3.1 Full tensor (i.e. full expected shape)
            (1, 45, 17),                    #27     # 3.1 Full tensor (i.e. full expected shape)
            (1, 1, 23),                     #28     # 3.2 Tensor reduce on one or more dims to 1
            (1, 64, 1),                     #29     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100),                  #30     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000, 100),                 #31     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000),                  #32     # 4.4 Extreme ratios between height/width
            (1, 9920, 1),                   #33     # 4.4 Extreme ratios between height/width  
            (1, 10000, 1),                  #34     # 4.4 Extreme ratios between height/width 
            (1, 32, 64),                    #35     # 4.1 Divisible by 32
            (1, 160, 96),                   #36     # 4.1 Divisible by 32
            (1, 17, 41),                    #37     # 4.2 Prime numbers
            (1, 89, 3),                     #38     # 4.2 Prime numbers

        # 3 dim with microbatch size > 1
            (2, 3, 4),                      #39     # 3.1 Full tensor (i.e. full expected shape)   
            (11, 45, 17),                   #40     # 3.1 Full tensor (i.e. full expected shape)
            (11, 1, 23),                    #41     # 3.2 Tensor reduce on one or more dims to 1
            (11, 64, 1),                    #42     # 3.2 Tensor reduce on one or more dims to 1
            (100, 100, 100),                #43     # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000, 100),                #44     # 4.3 Very large (thousands, 10s of thousands)
            (2, 10, 1000),                  #45     # 4.4 Extreme ratios between height/width
            (2, 9920, 1),                   #46     # 4.4 Extreme ratios between height/width  
            (10, 10000, 1),                 #47     # 4.4 Extreme ratios between height/width 
            (32, 32, 64),                   #48     # 4.1 Divisible by 32
            (64, 160, 96),                  #49     # 4.1 Divisible by 32
            (11, 17, 41),                   #50     # 4.2 Prime numbers
            (13, 89, 3),                    #51     # 4.2 Prime numbers

        # 4 dim with microbatch size = 1
            (1, 2, 3, 4),                   #52     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 45, 17),                #53     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 1, 23),                 #54     # 3.2 Tensor reduce on one or more dims to 1
            (1, 11, 64, 1),                 #55     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100, 100),             #56     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000, 100),             #57     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1, 10, 1000),               #58     # 4.4 Extreme ratios between height/width
            (1, 1, 9920, 1),                #59     # 4.4 Extreme ratios between height/width  
            (1, 10, 10000, 1),              #60     # 4.4 Extreme ratios between height/width 
            (1, 32, 32, 64),                #61     # 4.1 Divisible by 32
            (1, 64, 160, 96),               #62     # 4.1 Divisible by 32
            (1, 11, 17, 41),                #63     # 4.2 Prime numbers
            (1, 13, 89, 3),                 #64     # 4.2 Prime numbers

        # 4 dim with microbatch size > 1
            (3, 11, 45, 17),                #65     # 3.1 Full tensor (i.e. full expected shape)                  
            (2, 2, 3, 4),                   #66     # 3.1 Full tensor (i.e. full expected shape)  
            (4, 11, 1, 23),                 #67     # 3.2 Tensor reduce on one or more dims to 1  
            (5, 11, 64, 1),                 #68     # 3.2 Tensor reduce on one or more dims to 1  
            (6, 100, 100, 100),             #69     # 4.3 Very large (thousands, 10s of thousands)      
            (7, 10, 1000, 100),             #70     # 4.3 Very large (thousands, 10s of thousands)      
            (8, 1, 10, 1000),               #71     # 4.4 Extreme ratios between height/width      
            (9, 1, 9920, 1),                #72     # 4.4 Extreme ratios between height/width        
            (10, 10, 10000, 1),             #73     # 4.4 Extreme ratios between height/width       
            (11, 32, 32, 64),               #74     # 4.1 Divisible by 32  
            #Fatal Python error: Segmentation fault 
            pytest.param((12, 64, 160, 96), marks=pytest.mark.skip(reason=FailingReasons.SEG_FAULT)), #75     # 4.1 Divisible by 32          
            (13, 11, 17, 41),               #76     # 4.2 Prime numbers      
            (14, 13, 89, 3),                #77     # 4.2 Prime numbers      
    ]



########## HELPER METHOD USED FOR ERROR SUMMARY 

def xfail_test(input_operator, input_shape, input_model, input_kwargs):
    s = get_input_shapes()
    micro_batch_size = input_shape[0]
    match input_operator:
        case "Argmax":
            if(len(input_shape) == 2 and micro_batch_size > 1 and input_model in ("model_op_src_from_another_op", "model_op_src_from_tm_edge2")):
                # E           AssertionError: Error during inference
                pytest.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED)
            elif(input_shape in ((s[16],) + (s[20],) + (s[21],)) and input_model == "model_op_src_from_tm_edge1"):
                # E           AssertionError: Error during inference
                pytest.xfail(reason=FailingReasons.BUGGY_SHAPE)
            elif(input_shape in ((s[31],) + (s[33],) + (s[36],) + (s[44],) + (s[46],) + (s[49],)+ (s[56],) + (s[57],) + (s[59],) + tuple(s[60:63]) + (s[69],) + (s[70],) + tuple(s[72:75]))):
                # E           RuntimeError: 1/2/3 Nodes have no valid grids, exiting
                pytest.xfail(reason=FailingReasons.BUGGY_SHAPE)
        case "Dropout":
            # Error message: E       AssertionError: Data mismatch detected
            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
        case "LogicalNot":
            # Error message: E               KeyError: 'logical_not'
            pytest.xfail(reason=FailingReasons.NOT_IMPLEMENTED)
        case "Tilize":
            # Error message: E       AttributeError: module 'torch' has no attribute 'tensors'
            pytest.xfail(reason=FailingReasons.INFERENCE_FAILED)
        case "CumSum":
            if input_model in ("model_op_src_from_dram", "model_op_src_from_host", "model_op_src_from_another_op"):
                # E               RuntimeError: Input operand not mapped to new graph during lowering: CumSum1
                pytest.xfail(reason=FailingReasons.COMPILATION_FAILED)
            elif input_model in ("model_op_src_const_inputs1", "model_op_src_from_tm_edge1", "model_op_src_from_tm_edge2"):
                # E               RuntimeError: TT_ASSERT @ pybuda/csrc/passes/lowering_context.cpp:28: old_node->node_type() != graphlib::NodeType::kPyOp
                pytest.xfail(reason=FailingReasons.COMPILATION_FAILED)
        case "Pow":
            if(micro_batch_size > 1):
                if(input_kwargs['exponent'] not in (1000, 10000) and len(input_shape) == 2):
                    # E           AssertionError: Error during inference
                    pytest.xfail(reason=FailingReasons.INFERENCE_FAILED)
                elif(input_kwargs['exponent'] == 1000):
                    if(input_shape in (tuple(s[13:26]))):
                        # E           AssertionError: Error during inference
                        pytest.xfail(reason=FailingReasons.INFERENCE_FAILED)
                    elif(input_model in ("model_op_src_from_host", "model_op_src_from_tm_edge1", "model_op_src_from_dram") and input_shape in ((s[39],) + (s[41],) + (s[66],))):
                        # E           AssertionError: Data mismatch detected
                        pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                    elif(input_model in ("model_op_src_const_inputs1") and input_shape in (s[39],)):
                        # E           AssertionError: Data mismatch detected
                        pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                elif(input_kwargs['exponent'] == 10000):
                    if(input_shape in (tuple(s[13:26]))):
                        # E           AssertionError: Error during inference
                        pytest.xfail(reason=FailingReasons.INFERENCE_FAILED)
                    elif(input_model in ("model_op_src_from_host", "model_op_src_from_tm_edge1", "model_op_src_from_dram") and input_shape in (tuple(s[39:52]) + tuple(s[65:69]) + tuple(s[71:75]) + tuple(s[76:78]))):
                        # E           AssertionError: Data mismatch detected
                        pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                    elif(input_model in ("model_op_src_const_inputs1") and input_shape in ((s[39],) + (s[41],) + (s[66],))):
                        # E           AssertionError: Data mismatch detected
                        pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
            else:
                match input_model:
                    case "model_op_src_from_host":
                        if (input_kwargs['exponent'] == 1000 and input_shape in (tuple(s[0:5]) + tuple(s[9:13]) + (s[26],) + (s[28],) + (s[29],) + (s[38],) + (s[52],) + (s[54],)) ):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                        elif (input_kwargs['exponent'] == 10000 and input_shape in (tuple(s[0:13]) + tuple(s[26:39]) + tuple(s[52:65]))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                    case "model_op_src_from_dram":
                        if (input_kwargs['exponent'] == 1000 and input_shape in (tuple(s[0:5]) + tuple(s[9:13]) + (s[26],) + (s[28],) + (s[29],) + (s[38],) + (s[52],) + (s[54],))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                        elif (input_kwargs['exponent'] == 10000 and input_shape in (tuple(s[0:13]) + tuple(s[26:39]) + tuple(s[52:65]))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                    case "model_op_src_const_inputs1":
                        if (input_kwargs['exponent'] == 160 and input_shape in (s[3],)):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                        elif (input_kwargs['exponent'] == 1000 and input_shape in (tuple(s[0:2]) + (s[3],) + (s[12],) + (s[26],))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                        elif (input_kwargs['exponent'] == 10000 and input_shape in (tuple(s[0:4]) + tuple(s[11:13]) + (s[26],) + (s[28],) + (s[52],))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                    case "model_op_src_from_tm_edge1":
                        if (input_kwargs['exponent'] == 1000 and input_shape in (tuple(s[0:5]) + tuple(s[9:13]) + (s[26],) + (s[28],) + (s[29],) + (s[38],) + (s[52],) + (s[54],))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                        elif (input_kwargs['exponent'] == 10000 and input_shape in (tuple(s[0:13]) + tuple(s[26:39]) + tuple(s[52:65]))):
                            # E           AssertionError: Data mismatch detected
                            pytest.xfail(reason=FailingReasons.DATA_MISMATCH)
                    case "model_op_src_from_another_op", "model_op_src_from_tm_edge2":
                        return
        case _:
            if(len(input_shape) == 2 and micro_batch_size > 1):
                # E           AssertionError: Error during inference
                pytest.xfail(reason=FailingReasons.INFERENCE_FAILED)





########## TEST ALL ELEMENT-WISE UNARY OPS

@pytest.mark.parametrize("input_shape", get_input_shapes())
@pytest.mark.parametrize("input_model", [item.split(".")[0] for item in os.listdir(TEST_PLAN_MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("input_operator", get_eltwise_unary_operators())
def test_eltwise_unary_ops_per_test_plan(
    input_operator,
    input_model,
    input_shape,
    test_device,
    input_dev_data_format=None,
    input_math_fidelity=None
):
    kwargs = {}
    if input_operator == "LeakyRelu":
        kwargs['alpha'] = np.random.rand()
    xfail_test(input_operator, input_shape, input_model, kwargs)
    verify(
        input_model = input_model,
        input_operator = input_operator,
        input_shape = input_shape,
        kwargs = kwargs,
        input_dev_data_format = input_dev_data_format,
        input_math_fidelity = input_math_fidelity, 
        test_device = test_device, 
    )
# 5556 passed, 108 skipped, 2760 xfailed, 1 warning in 1103.93s (0:18:23)




########## TEST ELEMENT-WISE UNARY OP - POW

def get_pow_kwargs():
    return [
        # Error message: E                RuntimeError: TT_ASSERT @ pybuda/csrc/graph_lib/shape.cpp:34: values.size() >= BUDA_DIM_COUNT and values.size() <= BUDA_MAX_DIM_COUNT
        # 18 are always xpassed
        pytest.param(0.9336911808323198,    marks=pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),
        0,
        1,
        2,
        3,
        47, 
        160, 
        1000, 
        10000, 
    ]
@pytest.mark.parametrize("input_shape", get_input_shapes())
@pytest.mark.parametrize("input_model", [item.split(".")[0] for item in os.listdir(TEST_PLAN_MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("input_operator", ["Pow"])
@pytest.mark.parametrize("input_kwargs", get_pow_kwargs())
def test_eltwise_unary_ops_per_test_plan_pow(
    input_kwargs,
    input_operator,
    input_model,
    input_shape,
    test_device,
    input_dev_data_format=None,
    input_math_fidelity=None
):
    kwargs = {}
    kwargs['exponent'] = input_kwargs
    xfail_test(input_operator, input_shape, input_model, kwargs)
    verify(
        input_model = input_model,
        input_operator = input_operator,
        input_shape = input_shape,
        kwargs = kwargs,
        input_dev_data_format = input_dev_data_format,
        input_math_fidelity = input_math_fidelity, 
        test_device = test_device, 
    )
# 2813 passed, 54 skipped, 1327 xfailed, 18 xpassed in 526.40s (0:08:46) 




########## TEST ELEMENT-WISE UNARY OP - CLIP

def get_clip_kwargs():
    return [
        # min < max
        (0.4992656851851959, 0.9336911808323198),
        # min > max
        (0.9336911808323198, 0.4992656851851959),
        (0.4992656851851959, None),
        (None, 0.9336911808323198),
        # Error message: E               RuntimeError: yaml-cpp: error at line 22, column 70: bad conversion
        # Error message: E               RuntimeError: Unexpected index
        pytest.param(None, None,    marks=pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),
    ]
@pytest.mark.parametrize("input_shape", get_input_shapes())
@pytest.mark.parametrize("input_model", [item.split(".")[0] for item in os.listdir(TEST_PLAN_MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("input_operator", ["Clip"])
@pytest.mark.parametrize("input_kwargs_min, input_kwargs_max", get_clip_kwargs())
def test_eltwise_unary_ops_per_test_plan_clip(
    input_kwargs_min,
    input_kwargs_max,
    input_operator,
    input_model,
    input_shape,
    test_device,
    input_dev_data_format=None,
    input_math_fidelity=None
):
    kwargs = {}
    kwargs['min'] = input_kwargs_min    
    kwargs['max'] = input_kwargs_max
    xfail_test(input_operator, input_shape, input_model, kwargs)
    verify(
        input_model = input_model,
        input_operator = input_operator,
        input_shape = input_shape,
        kwargs = kwargs,
        input_dev_data_format = input_dev_data_format,
        input_math_fidelity = input_math_fidelity, 
        test_device = test_device, 
    )




########## TEST ELEMENT-WISE UNARY OP - CumSum

def get_cum_sum_kwargs_exclusive():
    return [
        False,
        # Error message:E   Assertion error: Currently not supported
        pytest.param(True,    marks=pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_PARAMETER_VALUE)),
    ]
@pytest.mark.parametrize("input_shape", get_input_shapes())
@pytest.mark.parametrize("input_model", [item.split(".")[0] for item in os.listdir(TEST_PLAN_MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("input_operator", ["CumSum"])
@pytest.mark.parametrize("input_kwargs_exclusive", get_cum_sum_kwargs_exclusive())
def test_eltwise_unary_ops_per_test_plan_cum_sum(
    input_kwargs_exclusive,
    input_operator,
    input_model,
    input_shape,
    test_device,
    input_dev_data_format=None,
    input_math_fidelity=None
):
    kwargs = {}
    kwargs['axis'] = np.random.randint(0, len(input_shape))
    kwargs['exclusive'] = input_kwargs_exclusive
    xfail_test(input_operator, input_shape, input_model, kwargs)
    verify(
        input_model = input_model,
        input_operator = input_operator,
        input_shape = input_shape,
        kwargs = kwargs,
        input_dev_data_format = input_dev_data_format,
        input_math_fidelity = input_math_fidelity, 
        test_device = test_device, 
    )
# 12 skipped, 924 xfailed in 11.99s




########## TEST ELEMENT-WISE UNARY OP - Dropout/LogicalNot/Tilize
### tests for this ops are always failing, because of that, they are tested only in single combination of model and input shape
### when they are fixed (bug reports: #2803, #2590, #2593) uncomment those 3 ops in function 'get_eltwise_unary_operators' and then run test named 'test_eltwise_unary_ops_per_test_plan' with flags: --runxfail --no-skips, so they can be tested in all needed combinations and uncomment next line:
# @pytest.mark.skip
@pytest.mark.parametrize("input_shape", [(1, 4)])
@pytest.mark.parametrize("input_model", ["model_op_src_from_host"])
@pytest.mark.parametrize("input_operator", ["Dropout", "LogicalNot", "Tilize"])
def test_eltwise_unary_ops_per_test_plan_droput_logicalnot_tilize(
    input_operator,
    input_model,
    input_shape,
    test_device,
    input_dev_data_format=None,
    input_math_fidelity=None
):
    kwargs = {}
    xfail_test(input_operator, input_shape, input_model, kwargs)
    verify(
        input_model = input_model,
        input_operator = input_operator,
        input_shape = input_shape,
        kwargs = kwargs,
        input_dev_data_format = input_dev_data_format,
        input_math_fidelity = input_math_fidelity, 
        test_device = test_device, 
    )





########## TEST DATA FORMAT AND MATH FIDELITY FOR ALL ELEMENT-WISE UNARY OPS

# We will not test all combinations of Data Format and Math Fidelity because it would be too much tests. 
#   1. First we will choose Data Format to be Float16_b and test all Math Fidelity values
#   2. Then we will set Math Fidelity to HiFi4 and test all Data Formats. 

def get_input_shape():
    return  (1, 45, 17)     #0     # 3.1 Full tensor (i.e. full expected shape)

dev_data_formats = [
    pybuda.DataFormat.Float16_b,
]

compiler_math_fidelity = [
                            pybuda.MathFidelity.LoFi,
                            pybuda.MathFidelity.HiFi2,
                            pybuda.MathFidelity.HiFi3,
                            pybuda.MathFidelity.HiFi4,
                         ]

@pytest.mark.parametrize("input_operator", get_eltwise_unary_operators())
@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_ops_mf_inputs(input_operator, test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan(input_operator, "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
#  60 passed, 12 xfailed in 8.12s 

@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_op_pow_mf_inputs(test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan_pow(1, "Pow", "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
# 4 passed in 1.55s 

@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_op_clip_mf_inputs(test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan_clip(np.random.rand(), np.random.rand(), "Clip", "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
# 4 passed in 1.67s 

@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_op_cum_sum_mf_inputs(test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan_cum_sum(False, "CumSum", "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
#  4 xfailed in 1.37s 



dev_data_formats=[
    pybuda.DataFormat.Bfp2,
    pybuda.DataFormat.Bfp2_b,
    pybuda.DataFormat.Bfp4,
    pybuda.DataFormat.Bfp4_b,
    pybuda.DataFormat.Bfp8,
    pybuda.DataFormat.Bfp8_b,
    pybuda.DataFormat.Float16,
    pybuda.DataFormat.Float16_b,
    pybuda.DataFormat.Float32,
    pybuda.DataFormat.Int8,
    pybuda.DataFormat.Lf8,
    pybuda.DataFormat.RawUInt16,
    pybuda.DataFormat.RawUInt32,
    pybuda.DataFormat.RawUInt8,
    pybuda.DataFormat.UInt16,
]

compiler_math_fidelity = [
    pybuda.MathFidelity.HiFi4,
]

@pytest.mark.parametrize("input_operator", get_eltwise_unary_operators())
@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_ops_df_inputs(input_operator, test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan(input_operator, "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
# 225 passed, 45 xfailed in 36.47s

@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_op_pow_df_inputs(test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan_pow(1, "Pow", "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
# 15 passed in 2.71s

@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_op_clip_df_inputs(test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan_clip(np.random.rand(), np.random.rand(), "Clip", "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
# 15 passed in 2.69s

@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_eltwise_unary_op_cum_sum_df_inputs(test_device, dev_data_format, math_fidelity):
    test_eltwise_unary_ops_per_test_plan_cum_sum(False, "CumSum", "model_op_src_from_host", get_input_shape(), test_device, dev_data_format, math_fidelity)
# 15 xfailed in 1.47s






########## SINGLE TEST FOR ALL ELEMENT-WISE UNARY OPS
# run only from command line
# used to reproduce bugs 

@pytest.mark.skip
def test_eltwise_unary_ops_per_test_plan_single(
        un_op,
        un_model,
        un_shape,
        test_device
):
    test_eltwise_unary_ops_per_test_plan(un_op, un_model, un_shape, test_device)

@pytest.mark.skip
def test_eltwise_unary_ops_per_test_plan_pow_single(
        un_model,
        un_shape,
        un_kwargs,
        test_device
):
    test_eltwise_unary_ops_per_test_plan_pow(un_kwargs['exponent'], "Pow", un_model, un_shape, test_device)

@pytest.mark.skip
def test_eltwise_unary_ops_per_test_plan_clip_single(
        un_model,
        un_shape,
        un_kwargs,
        test_device
):
    test_eltwise_unary_ops_per_test_plan_clip(un_kwargs['min'], un_kwargs['max'], "Clip", un_model, un_shape, test_device)

@pytest.mark.skip
def test_eltwise_unary_ops_per_test_plan_cum_sum_single(
        un_model,
        un_shape,
        un_kwargs,
        test_device
):
    test_eltwise_unary_ops_per_test_plan_cum_sum(un_kwargs['exclusive'], "CumSum", un_model, un_shape, test_device)











#######################################################################################

########## OLD TESTS 
# those tests are skipped

MODELS_PATH = "./pybuda/test/operators/eltwise_unary/models/"

SHAPE_NO = 2
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 5
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True

np.random.seed(1)

# tensors of arbitrarily shape
shape = [np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, np.random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)).tolist() for _ in range(SHAPE_NO)]
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


@pytest.mark.parametrize("shape", shape, ids=[f"shape{'x'.join([str(jtem) for jtem in item])}" for item in shape])
@pytest.mark.parametrize("operation", ["Abs", "LeakyRelu", "Exp", "Identity", "Reciprocal", "Sigmoid", "Sqrt", "Gelu", "Log", "Relu", "Buffer", "Tanh", "Dropout", "Sine", "Cosine", "Argmax", "Clip"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def obsoleted_test_eltwise_unary(
    op_test_kind,
    operation,
    model,
    shape
):
    test_kind = op_test_kind

    if model == "model_9" and operation == "Reciprocal":
        pytest.xfail("tenstorrent/pybuda#18")

    kwargs = {}
    pcc = 0.99
    if operation == "LeakyRelu":
        kwargs['alpha'] = np.random.rand()
        if test_kind.is_training():
            pcc = 0.95
    if operation == "Clip":
        kwargs['min'] = np.random.rand()
        kwargs['max'] = np.random.rand()
        
    architecture = f'models.{model}.BudaElementWiseUnaryTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape}'
    for k, v in kwargs.items():
        architecture = f'{architecture}, {k}={v}'
    architecture = f'{architecture})'
    
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)

    #Fusing disabled due to tenstorrent/pybuda#784
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute(),
                        enable_auto_fusing=False
                     ), 
        verify_cfg=VerifyConfig(pcc=pcc)
    )
