# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise binary operators
#
# In this test we use pytorch tensors and operators to verify buda operators



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
# (-) 7. Special attributes - if applicable.. like approx_mode for Exp, for example


import os
import pytest
import numpy as np

from typing import List, Dict, Type
from loguru import logger

import torch
import pybuda
import pybuda.op

from pybuda import PyBudaModule
from pybuda.op_repo import TensorShape
from test.operators.utils import netlist_utils, InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.conftest import TestDevice

from pybuda import TTDevice, pybuda_compile, VerifyConfig, CompilerConfig

# from . import models
# from .models import test_plan


class ModelFromAnotherOp(PyBudaModule):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_from_another_op")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        # we use Add and Subtract operators to create two operands which are inputs for the binary operator
        xx = pybuda.op.Add("Add0", x, y)
        yy = pybuda.op.Subtract("Subtract0", x, y)
        output = self.operator(self.opname + "1", xx, yy, **self.kwargs)
        return output


class ModelFromHost(PyBudaModule):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_from_host")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        output = self.operator(self.opname + "0", x, y, **self.kwargs)
        return output


class ModelFromDramQueue(PyBudaModule):

    model_name = "model_op_src_from_dram_queue"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_from_dram_queue")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_from_dram_queue"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        output = self.operator(self.opname + "0", x, y, **self.kwargs)
        return output


class ModelFromDramQueuePrologued(PyBudaModule):

    model_name = "model_op_src_from_dram_queue_prologued"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_from_dram_queue_prologued")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_from_dram_queue_prologued"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()

        self.shape_input = ShapeUtils.reduce_microbatch_size(shape)

        self.add_constant("c")
        self.set_constant("c", pybuda.Tensor.create_from_torch(my_rand(*self.shape_input), constant=True))

    def forward(self, x):
        output = self.operator(self.opname + "0", self.get_constant("c"), x, **self.kwargs)
        return output


class ModelConstEvalPass(PyBudaModule):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_const_eval_pass")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()
        
        self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)

        self.add_constant("c1")
        self.set_constant("c1", pybuda.Tensor.create_from_torch(my_rand(*self.constant_shape), constant=True))

        self.add_constant("c2")
        self.set_constant("c2", pybuda.Tensor.create_from_torch(my_rand(*self.constant_shape), constant=True))
       
        self.inputs = [
            pybuda.Tensor.create_from_torch(my_rand(*self.shape))
        ]

    def forward(self, x, y):
        v1 = self.operator(self.opname + "0", self.get_constant("c1"), self.get_constant("c2"), **self.kwargs)
        # v2 and v3 consume inputs
        v2 = pybuda.op.Add("Add1", x, y)
        v3 = pybuda.op.Add("Add2", v1, v2)
        return v3


class ModelOpSrcFromTmEdge1(PyBudaModule):

    model_name = "model_op_src_from_tm_edge1"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_from_tm_edge1")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_from_tm_edge1"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        xx = pybuda.op.Add("Add0", x, y)
        yy = pybuda.op.tm.Transpose("Transpose0", xx, -1, -2)
        output = self.operator(self.opname + "1", yy, yy, **self.kwargs)
        return output


class ModelOpSrcFromTmEdge2(PyBudaModule):

    model_name = "model_op_src_from_tm_edge2"

    def __init__(self, operator, opname, shape, kwargs):
        super().__init__("Element_wise_binary_operator_" + opname + "_test_op_src_from_tm_edge2")
        self.testname = "Element_wise_binary_operator_" + opname + "_test_op_src_from_tm_edge2"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        xx = pybuda.op.tm.Transpose("Transpose0", x, -1, -2)
        yy = pybuda.op.tm.Transpose("Transpose1", y, -1, -2)
        output = self.operator(self.opname + "2", xx, yy, **self.kwargs)
        return output


def verify(
    test_device: TestDevice,
    model_type: Type[PyBudaModule],
    input_operator: str,
    input_shape: TensorShape,
    number_of_operands: int,
    kwargs: Dict = {},
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: pybuda.DataFormat = None,
    math_fidelity: pybuda.MathFidelity = None,
):
    '''Common verification function for all tests'''

    operator = getattr(pybuda.op, input_operator)

    model = model_type(operator=operator, opname=input_operator, shape=input_shape, kwargs=kwargs)

    input_shapes = tuple([input_shape for _ in range(number_of_operands)])
    logger.trace(f"***input_shapes: {input_shapes}")

    VerifyUtils.verify(
        model=model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


MODEL_TYPES = [
    ModelFromAnotherOp,
    ModelFromHost,
    ModelFromDramQueue,
    # ModelFromDramQueuePrologued,
    ModelConstEvalPass,
    ModelOpSrcFromTmEdge1,
    ModelOpSrcFromTmEdge2,
]


def get_eltwise_binary_ops():
    return [
        "Add",              #00
        "Max",              #01
        "Min",              #02
        "Power",            #03
        "Subtract",         #04
        "Multiply",         #05
        "Heaviside",        #06
        "Greater",          #07
        "GreaterEqual",     #08
        "Less",             #09
        "LessEqual",        #10
        "Equal",            #11
        "NotEqual",         #12
        "Divide",           #13
        "BinaryStack",      #14
    ]

def get_input_shapes():
    return [
            # 2-dimensional shape, microbatch_size = 1:
            (1, 4),                     #00      # 3.1 Full tensor (i.e. full expected shape)
            (1, 17),                    #01      # 3.1 Full tensor (i.e. full expected shape)
            (1, 23),                    #02      # 3.2 Tensor reduce on one or more dims to 1
            (1, 1),                     #03      # 3.2 Tensor reduce on one or more dims to 1
            (1, 100),                   #04      # 4.3 Very large (thousands, 10s of thousands)
            (1, 500),                   #05      # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000),                  #06      # 4.4 Extreme ratios between height/width
            (1, 1920),                  #07      # 4.4 Extreme ratios between height/width
            (1, 10000),                 #08      # 4.4 Extreme ratios between height/width
            (1, 64),                    #09      # 4.1 Divisible by 32
            (1, 96),                    #10     # 4.1 Divisible by 32
            (1, 41),                    #11     # 4.2 Prime numbers
            (1, 3),                     #12     # 4.2 Prime numbers
            
            # 2-dimensional shape, microbatch_size > 1:
            # All shapes fails for all operators
            pytest.param((3, 4),        #13      # 3.1 Full tensor (i.e. full expected shape)
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((45, 17),      #14      # 3.1 Full tensor (i.e. full expected shape)
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((64, 1),       #15      # 3.2 Tensor reduce on one or more dims to 1
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((100, 100),    #16      # 4.3 Very large (thousands, 10s of thousands)
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((1000, 100),   #17      # 4.3 Very large (thousands, 10s of thousands)
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((10, 1000),    #18      # 4.4 Extreme ratios between height/width
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((9920, 1),     #19      # 4.4 Extreme ratios between height/width  
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((10000, 1),    #20      # 4.4 Extreme ratios between height/width 
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((32, 64),      #21      # 4.1 Divisible by 32
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((160, 96),     #22      # 4.1 Divisible by 32
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((17, 41),      #23      # 4.2 Prime numbers
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),
            pytest.param((89, 3),       #24      # 4.2 Prime numbers
                         marks=pytest.mark.xfail(reason="Skip shapes where microbatchsize > 1")),

            # 3-dimensional shape, microbatch_size = 1:
            (1, 3, 4),                  #25     # 3.1 Full tensor (i.e. full expected shape)
            (1, 45, 17),                #26     # 3.1 Full tensor (i.e. full expected shape)
            (1, 1, 23),                 #27     # 3.2 Tensor reduce on one or more dims to 1
            (1, 64, 1),                 #28     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100),              #29     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000, 100),             #30     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000),              #31     # 4.4 Extreme ratios between height/width
            (1, 9920, 1),               #32     # 4.4 Extreme ratios between height/width
            (1, 10000, 1),              #33     # 4.4 Extreme ratios between height/width 
            (1, 32, 64),                #34     # 4.1 Divisible by 32
            (1, 160, 96),               #35     # 4.1 Divisible by 32
            (1, 17, 41),                #36     # 4.2 Prime numbers
            (1, 89, 3),                 #37     # 4.2 Prime numbers

            # 3-dimensional shape, microbatch_size > 1:
            (2, 3, 4),                  #38     # 3.1 Full tensor (i.e. full expected shape)
            (11, 45, 17),               #39     # 3.1 Full tensor (i.e. full expected shape)
            (11, 1, 23),                #40     # 3.2 Tensor reduce on one or more dims to 1
            (11, 64, 1),                #41     # 3.2 Tensor reduce on one or more dims to 1
            (100, 100, 100),            #42     # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000, 100),            #43     # 4.3 Very large (thousands, 10s of thousands)
            (10, 10000, 1),             #44     # 4.4 Extreme ratios between height/width
            (32, 32, 64),               #45     # 4.1 Divisible by 32
            (64, 160, 96),              #46     # 4.1 Divisible by 32
            (11, 17, 41),               #47     # 4.2 Prime numbers
            (13, 89, 3),                #48     # 4.2 Prime numbers

            # 4-dimensional shape, microbatch_size = 1:
            (1, 2, 3, 4),               #49     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 45, 17),            #50     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 1, 23),             #51     # 3.2 Tensor reduce on one or more dims to 1
            (1, 11, 64, 1),             #52     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100, 100),         #53     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000, 100),         #54     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1, 10, 1000),           #55     # 4.4 Extreme ratios between height/width
            (1, 1, 9920, 1),            #56     # 4.4 Extreme ratios between height/width
            (1, 10, 10000, 1),          #57     # 4.4 Extreme ratios between height/width
            (1, 32, 32, 64),            #58     # 4.1 Divisible by 32
            (1, 64, 160, 96),           #59     # 4.1 Divisible by 32
            (1, 11, 17, 41),            #60     # 4.2 Prime numbers
            (1, 13, 89, 3),             #61     # 4.2 Prime numbers

            # 4-dimensional shape, microbatch_size > 1:
            (3, 11, 45, 17),                  #62     # 3.1 Full tensor (i.e. full expected shape)
            (2, 2, 3, 4),                     #63     # 3.1 Full tensor (i.e. full expected shape)
            (4, 11, 1, 23),                   #64     # 3.2 Tensor reduce on one or more dims to 1
            (5, 11, 64, 1),                   #65     # 3.2 Tensor reduce on one or more dims to 1
            (6, 100, 100, 100),               #66     # 4.3 Very large (thousands, 10s of thousands)
            (7, 10, 1000, 100),               #67     # 4.3 Very large (thousands, 10s of thousands)
            (8, 1, 10, 1000),                 #68     # 4.4 Extreme ratios between height/width
            (9, 1, 9920, 1),                  #69     # 4.4 Extreme ratios between height/width
            (10, 10, 10000, 1),               #70     # 4.4 Extreme ratios between height/width
            (11, 32, 32, 64),                 #71     # 4.1 Divisible by 32
            pytest.param((12, 64, 160, 96),   #72     # 4.1 Divisible by 32
                         marks=pytest.mark.skip(reason="RuntimeError: Fatal Python error: Segmentation fault")),
            (13, 11, 17, 41),                 #73     # 4.2 Prime numbers
            (14, 13, 89, 3),                  #74     # 4.2 Prime numbers
    ]


@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("input_shape", get_input_shapes())
def test_eltwise_binary_ops_per_test_plan(
    input_operator,
    model_type,
    input_shape,
    test_device,
    dev_data_format=None, 
    input_math_fidelity=None
):
    s = get_input_shapes()
    
    # Observed Bugs: --------------------------------------------------------------------------------------------------------------------
    # 1. input_shape in ((1, 1000, 100), (10, 1000, 100)):
    if model_type == ModelOpSrcFromTmEdge1 and input_operator == "Heaviside" and input_shape in (s[30], s[43]):
        # Error Message: "RuntimeError: TT_ASSERT @ pybuda/csrc/balancer/policies/policy_utils.cpp:2221: " + 
        #                "graph ->get_edges( graph->get_node_by_name(nopInsertInst->src), " +
        #                "graph->get_node_by_name(nopInsertInst->dest)) .size() == 1"
        pytest.xfail(reason="Buggy shapes for ModelOpSrcFromTmEdge1.")
    # 2. input_shape in ((1, 9920, 1), (1, 1, 9920, 1), (9, 1, 9920, 1)):
    if model_type == ModelFromAnotherOp and input_operator in ["Equal", "NotEqual"] and input_shape in (s[32], s[56], s[69]):
        # Error Mesage: "RuntimeError: Fatal balancer error: Could not reconcile constraints: path[Add0 -> _fused_op_0]"
        pytest.xfail(reason="Buggy shapes for ModelFromAnotherOp.")
    # 3. BinaryStack bugs:
    if input_operator == "BinaryStack":
        if len(input_shape) in (2, 3):
            # input_shapes are 2-dimensional and 3-dimensional:
            pytest.xfail(reason="BinaryStack operator is not working for 2D and 3D shapes.")
        elif model_type == ModelConstEvalPass:
            # model_type is ModelConstEvalPass:
            pytest.xfail(reason="BinaryStack operator is not working for ModelConstEvalPass.")
        elif input_shape in (s[55], s[56], s[57], s[68], s[69], s[70]):
            # input_shapes are all with extreme ratios between height/width:
            pytest.xfail(reason="BinaryStack operator is not working for shapes that have extreme ratios between height/width")
    # ------------------------------------------------------------------------------------------------------------------------------------


    input_source_flag = None
    if model_type == ModelFromDramQueue:
        input_source_flag = InputSourceFlags.FROM_DRAM

    kwargs={}
    if input_operator == "BinaryStack":
        kwargs['dim'] = -1

    verify(
        test_device=test_device,
        model_type=model_type,
        input_operator=input_operator,
        input_shape=input_shape,
        number_of_operands=2,
        kwargs=kwargs,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=input_math_fidelity,
    )

    # netlist validations:

    file_path = VerifyUtils.get_netlist_filename()

    if model_type == ModelFromDramQueue:
        assert netlist_utils.read_netlist_value(file_path, "/queues/x/loc") == 'dram'
        assert netlist_utils.read_netlist_value(file_path, "/queues/y/loc") == 'dram'

    if model_type == ModelConstEvalPass:
        # Here we check there is no key with operator name in the netlist in graphs section
        d = netlist_utils.read_netlist_value(file_path, "/graphs/fwd_0_0_temporal_epoch_0")
        for key in d.keys():
            assert input_operator not in key


def get_eltwise_binary_ops_prologued():
    return [
        pytest.param("Add"),              #00
        pytest.param("Max"),              #01
        pytest.param("Min"),              #02
        pytest.param("Power",             #03
                     marks=pytest.mark.xfail(reason="Validation error caused by pcc threshold.")),
        pytest.param("Subtract"),         #04
        pytest.param("Multiply"),         #05
        pytest.param("Heaviside"),        #06
        pytest.param("Greater"),          #07
        pytest.param("GreaterEqual"),     #08
        pytest.param("Less"),             #09
        pytest.param("LessEqual"),        #10
        pytest.param("Equal"),            #11
        pytest.param("NotEqual"),         #12
        pytest.param("Divide"),           #13
        pytest.param("BinaryStack"),      #14
    ]

def get_input_shapes_prologued():
    # Columns: input_shape, input_source_flag, should_prolog"
    return [
            # 2-dimensional shape, microbatch_size = 1:
            ((1, 16),        InputSourceFlags.FROM_DRAM_PROLOGUED, True),                  #00        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 17),        InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False),             #01        # 3.1 Full tensor (i.e. full expected shape)
            
            # 2-dimensional shape, microbatch_size > 1:
            pytest.param((4, 16), InputSourceFlags.FROM_DRAM_PROLOGUED, True,              #02        # 3.1 Full tensor (i.e. full expected shape)
                    marks=pytest.mark.xfail(reason="Doesn't work for microbatchsize > 1 and two dimensions.")),
            pytest.param((3, 17), InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False,         #03        # 3.1 Full tensor (i.e. full expected shape)
                    marks=pytest.mark.xfail(reason="Doesn't work for microbatchsize > 1 and two dimensions.")),
            
            # 3-dimensional shape:
            ((2, 3, 3),      InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False),             #04        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUED, True),                  #05        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #06        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 3, 3),      InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False),             #07        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUED, True),                  #08        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #09 !!!    # 3.1 Full tensor (i.e. full expected shape) - not according to documentation!
            ((2, 10, 5),     InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #10        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 1, 15),     InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #11        # 3.2 Tensor reduce on one or more dims to 1
            ((2, 50, 1),     InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #12        # 3.2 Tensor reduce on one or more dims to 1
            ((2, 100, 100),  InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #13        # 4.3 Very large (thousands, 10s of thousands)
            ((2, 100, 1000), InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False),  #14        # 4.3 Very large (thousands, 10s of thousands)
            ((2, 1, 10000),  InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False),  #15        # 4.4 Extreme ratios between height/width
            ((2, 10000, 1),  InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False),  #16        # 4.4 Extreme ratios between height/width
            ((2, 32, 32),    InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #17        # 4.1 Divisible by 32
            ((2, 96, 96),    InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #18        # 4.1 Divisible by 32
            ((2, 13, 97),    InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),   #19        # 4.2 Prime numbers
            
            # 4-dimensional shape, microbatch_size = 1:
            ((1, 2, 3, 4),   InputSourceFlags.FROM_DRAM_PROLOGUED, True),                  #20        # 3.1 Full tensor (i.e. full expected shape)
            ((1, 17, 13, 4), InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False),             #21        # 3.1 Full tensor (i.e. full expected shape)
            
            # 4-dimensional shape, microbatch_size > 1:
            ((2, 2, 3, 4),   InputSourceFlags.FROM_DRAM_PROLOGUED, True),                  #22        # 3.1 Full tensor (i.e. full expected shape)
            ((2, 17, 13, 4), InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False),             #23        # 3.1 Full tensor (i.e. full expected shape)
            ]


@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops_prologued())
@pytest.mark.parametrize("model_type", [ModelFromDramQueuePrologued])
@pytest.mark.parametrize("input_shape, input_source_flag, should_prolog", get_input_shapes_prologued())
def test_eltwise_binary_ops_per_test_plan_dram_prologued(
    input_operator,
    model_type,
    input_shape,
    input_source_flag,
    should_prolog,
    test_device,
    dev_data_format=None,
    input_math_fidelity=None
):

    # Observed Bugs: --------------------------------------------------------------------------------------------------------------------
    # 1. BinaryStack bugs:
    if input_operator == "BinaryStack" and len(input_shape) in (2, 3):
        # input_shapes are 2-dimensional and 3-dimensional:
        pytest.xfail(reason="BinaryStack operator is not working for 2D and 3D shapes.")
    # -----------------------------------------------------------------------------------------------------------------------------------

    # Divide behaves differently from another operators for this shape
    if input_operator == "Divide" and input_shape == (2, 100, 1000):
        should_prolog = True

    kwargs = {}
    if input_operator == "BinaryStack":
        kwargs['dim'] = -1

    verify(
        test_device=test_device,
        model_type=model_type,
        input_operator=input_operator,
        input_shape=input_shape,
        number_of_operands=1,
        kwargs=kwargs,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=input_math_fidelity,
    )

    # netlist validation:
    file_path = VerifyUtils.get_netlist_filename()
    d = netlist_utils.read_netlist_value(file_path, "/programs/0/run_fwd_0/4/execute/queue_settings/input_0_" + input_operator + "0")
    if should_prolog:
        assert d['prologue']
    else:
        assert not d['prologue']


# Operand Data Format (DF) and Math Fidelity (MF)
# We will not test all combinations of Data Format and Math Fidelity
# because it would be too much tests. 
# Also, we will test DF and MF by fixing single shape.
#
#   1. First we will choose Data Format to be Float16_b and test all Math Fidelity values
#   2. Then we will set Math Fidelity to HiFi4 and test all Data Formats. 

### 1. ####################################################################################


def get_single_shape(microbatch_size=1):
    return (microbatch_size, 3, 3)        # Full tensor, small size

#   5.4 Operand DFs

dev_data_formats = [
    pybuda.DataFormat.Float16_b,
]

#  6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
compiler_math_fidelity = [
                            pybuda.MathFidelity.LoFi,       #00
                            pybuda.MathFidelity.HiFi2,      #01
                            pybuda.MathFidelity.HiFi3,      #02
                            pybuda.MathFidelity.HiFi4,      #03
                         ]


@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("model_type", [ModelFromAnotherOp])
@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_mf_eltwise_binary_ops_per_test_plan(input_operator, model_type, test_device, dev_data_format, math_fidelity):
    test_eltwise_binary_ops_per_test_plan(
        input_operator,
        model_type,
        get_single_shape(),
        test_device,
        dev_data_format,
        math_fidelity,
    )


### 2. ####################################################################################

#   5.4 Operand DFs

dev_data_formats=[
    pybuda.DataFormat.Bfp2,         #00
    pybuda.DataFormat.Bfp2_b,       #01
    pybuda.DataFormat.Bfp4,         #02
    pybuda.DataFormat.Bfp4_b,       #03
    pybuda.DataFormat.Bfp8,         #04
    pybuda.DataFormat.Bfp8_b,       #05
    pybuda.DataFormat.Float16,      #06
    pybuda.DataFormat.Float16_b,    #07
    pybuda.DataFormat.Float32,      #08
    pybuda.DataFormat.Int8,         #09
    pybuda.DataFormat.Lf8,          #10
    pybuda.DataFormat.RawUInt16,    #11
    pybuda.DataFormat.RawUInt32,    #12
    pybuda.DataFormat.RawUInt8,     #13
    pybuda.DataFormat.UInt16,       #14
]

#  6. Math fidelity
compiler_math_fidelity = [
    pybuda.MathFidelity.HiFi4,
]


@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("model_type", [ModelFromAnotherOp])
@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_df_eltwise_binary_ops_per_test_plan(input_operator, model_type, test_device, dev_data_format, math_fidelity):
    test_eltwise_binary_ops_per_test_plan(
        input_operator,
        model_type,
        get_single_shape(),
        test_device,
        dev_data_format,
        math_fidelity,
    )


# LogicalAnd operator:
# Compile is failing, looks like it is not supported by the compiler yet.
# Error Message: "Compile error: 'logical_and'"
# ...
# Error Message: "KeyError: 'logical_and'"
@pytest.mark.xfail(reason="Not implemented")
def test_eltwise_binary_logicaland_operator(test_device):

    verify(
        test_device=test_device,
        model_type=ModelFromHost,
        input_operator="LogicalAnd",
        input_shape=[1, 3, 3],
        number_of_operands=2,
    )


# It is not clear what the operator should do, because the documentation is missing - it is copied from Max operator.
# Case with dim=-1 is covered with other operators in test "test_eltwise_binary_ops_per_test_plan".
# This test covers all other values for dim parameter.
@pytest.mark.xfail(reason="Operator is not working for dim parameter different than -1.")
@pytest.mark.parametrize("shape", [(1, 3, 3, 3)])
@pytest.mark.parametrize("dim", [-2, 0, 1, 2])
@pytest.mark.parametrize("model", [ModelFromHost, ModelFromAnotherOp])
def test_eltwise_binary_binarystack_operator(test_device, shape, dim, model):

    kwargs={}
    kwargs['dim'] = dim

    verify(
        test_device=test_device,
        model_type=model,
        input_operator="BinaryStack",
        input_shape=shape,
        number_of_operands=2,
        kwargs=kwargs,
    )


# Test function for running single operator test with specific parameters
# with all models except prologued
@pytest.mark.skip
def test_eltwise_binary_ops_per_test_plan_single(
        bin_op,
        bin_model,
        bin_shape,
        test_device
):

    model = eval(bin_model)
    shape = eval(bin_shape) if type(bin_shape) is str else bin_shape
    
    test_eltwise_binary_ops_per_test_plan(bin_op, model, shape, test_device)


# Test function for running single operator test with specific parameters
# with prologued model
@pytest.mark.skip
def test_eltwise_binary_ops_per_test_plan_single_prologued(
        bin_op,
        bin_shape_prologued,
        test_device
):
    model = ModelFromDramQueuePrologued
    shape, source_flag, should_prolog = eval(bin_shape_prologued)

    test_eltwise_binary_ops_per_test_plan_dram_prologued(bin_op, model, shape, source_flag, should_prolog, test_device)


# ------------------------------------------------------------------------------------------------------------
# Old test implementation using not simplified test models:
# (These old tests are deactivated)
# ------------------------------------------------------------------------------------------------------------

MODELS_PATH = "./pybuda/test/operators/eltwise_binary/models/"

SHAPE_NO = 2
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 6
SHAPE_WDIM_MIN = 2
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True

np.random.seed(2)

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
@pytest.mark.parametrize("operation", ["Add", "Max", "Min", "Power", "Subtract", "Multiply", "Heaviside", "Greater", "GreaterEqual", "Less", "LessEqual", "Equal", "NotEqual"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("mode", ["Inference"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
def obsoleted_test_eltwise_binary(
    mode,
    recompute,
    operation,
    model,
    shape,
    test_device
):

    training = (mode == "Training")

    if training and (operation in ["Greater", "GreaterEqual", "Less", "LessEqual", "Equal", "NotEqual"]):
        pytest.skip("Comparison operators shouldn't have derivative, and backward.")

    if training and operation == "Heaviside":
        pytest.skip("Heaviside function shouldn't have derivative, and backward.")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaElementWiseBinaryTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)

    #Fusing disabled due to tenstorrent/pybuda#548
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute,
                        enable_auto_fusing=False
                     ), 
        verify_cfg=VerifyConfig()
    )
