# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise binary operators
#
# In this test we test pytorch binary operators


import pytest

from typing import List, Dict, Type
from loguru import logger

import random
import torch
import pybuda
import pybuda.op

from pybuda.op_repo import TensorShape
from test.operators.utils import netlist_utils, InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import FailingReasons
from test.conftest import TestDevice
from test.random.rgg import RateLimiter


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x.retain_grad()
        y.retain_grad()
        # we use Add and Subtract operators to create two operands which are inputs for the binary operator
        xx = torch.add(x, y)
        yy = torch.sub(x, y)
        output = self.operator(xx, yy, **self.kwargs)
        return output


class ModelFromHost(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromHost, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x.retain_grad()
        y.retain_grad()
        output = self.operator(x, y, **self.kwargs)
        return output


class ModelFromDramQueue(torch.nn.Module):

    model_name = "model_op_src_from_dram_queue"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromDramQueue, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_dram_queue"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x.retain_grad()
        y.retain_grad()
        output = self.operator(x, y, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs
        
        self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)

        # self.c1 = torch.rand(*self.constant_shape)
        # self.c2 = torch.rand(*self.constant_shape)
        self.c1 = (torch.rand(*self.constant_shape, requires_grad=False) - 0.5).detach()
        self.c2 = (torch.rand(*self.constant_shape, requires_grad=False) - 0.5).detach()

    def forward(self, x, y):
        v1 = self.operator(self.c1, self.c2, **self.kwargs)
        # v2 and v3 consume inputs
        x.retain_grad()
        y.retain_grad()
        v2 = torch.add(x, y)
        v3 = torch.add(v1, v2)
        return v3


class ModelOpSrcFromTmEdge1(torch.nn.Module):

    model_name = "model_op_src_from_tm_edge1"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelOpSrcFromTmEdge1, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_tm_edge1"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        x.retain_grad()
        y.retain_grad()
        xx = torch.add(x, y)
        yy = torch.transpose(xx, -1, -2)
        output = self.operator(yy, yy, **self.kwargs)
        return output


class ModelOpSrcFromTmEdge2(torch.nn.Module):

    model_name = "model_op_src_from_tm_edge2"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelOpSrcFromTmEdge2, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_tm_edge2"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x, y):
        x.retain_grad()
        y.retain_grad()
        xx = torch.transpose(x, -1, -2)
        yy = torch.transpose(y, -1, -2)
        output = self.operator(xx, yy, **self.kwargs)
        return output


def verify(
    test_device: TestDevice,
    model_type: Type[torch.nn.Module],
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

    operator = getattr(torch, input_operator)

    pytorch_model = model_type(operator=operator, opname=input_operator, shape=input_shape, kwargs=kwargs)
    pybuda_model = pybuda.PyTorchModule(pytorch_model.model_name, pytorch_model)

    input_shapes = tuple([input_shape for _ in range(number_of_operands)])
    logger.trace(f"***input_shapes: {input_shapes}")

    VerifyUtils.verify(
        model=pybuda_model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


MODEL_TYPES = [
    # ModelFromAnotherOp,
    ModelFromHost,
    # ModelFromDramQueue,
    # ModelConstEvalPass,
    # ModelOpSrcFromTmEdge1,
    # ModelOpSrcFromTmEdge2,
]


def get_eltwise_binary_ops():
    return [
        "add",                      #00
        "div",                      #01
        "divide",                   #02     - Alias for div.
        "mul",                      #03
        "multiply",                 #04     - Alias for mul.
        "sub",                      #05
        "subtract",                 #06     - Alias for sub.
        "true_divide",              #07     - Alias for div with rounding_mode=None.
        "eq",                       #08
        "ne",                       #09
        "le",                       #10
        "ge",                       #11
        "greater",                  #12    - Alias for gt.
        "greater_equal",            #13    - Alias for ge.
        "gt",                       #14
        "less_equal",               #15    - Alias for le.
        "lt",                       #16
        "less",                     #17    - Alias for lt.
        "maximum",                  #18
        "minimum",                  #19
        "not_equal",                #20    - Alias for ne.
    ]

def get_input_shapes():
    return [
            # 2-dimensional shape, microbatch_size = 1:
            pytest.param((1, 4),              marks=pytest.mark.run_in_pp),  #00      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 17),             marks=pytest.mark.slow),       #01      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 23),             marks=pytest.mark.slow),       #02      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 1),              marks=pytest.mark.slow),       #03      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 100),            marks=pytest.mark.slow),       #04      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 500),            marks=pytest.mark.slow),       #05      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 1000),           marks=pytest.mark.slow),       #06      # 4.4 Extreme ratios between height/width
            pytest.param((1, 1920),           marks=pytest.mark.slow),       #07      # 4.4 Extreme ratios between height/width
            pytest.param((1, 10000),          marks=pytest.mark.slow),       #08      # 4.4 Extreme ratios between height/width
            pytest.param((1, 64),             marks=pytest.mark.run_in_pp),  #09      # 4.1 Divisible by 32
            pytest.param((1, 96),             marks=pytest.mark.slow),       #10      # 4.1 Divisible by 32
            pytest.param((1, 41),             marks=pytest.mark.slow),       #11      # 4.2 Prime numbers
            pytest.param((1, 3),              marks=pytest.mark.slow),       #12      # 4.2 Prime numbers

            # 2-dimensional shape, microbatch_size > 1:
            # All shapes fails for all operators
            pytest.param((3, 4),        #13      # 3.1 Full tensor (i.e. full expected shape)
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.run_in_pp]),
            pytest.param((45, 17),      #14      # 3.1 Full tensor (i.e. full expected shape)
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((64, 1),       #15      # 3.2 Tensor reduce on one or more dims to 1
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((100, 100),    #16      # 4.3 Very large (thousands, 10s of thousands)
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((1000, 100),   #17      # 4.3 Very large (thousands, 10s of thousands)
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((10, 1000),    #18      # 4.4 Extreme ratios between height/width
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((9920, 1),     #19      # 4.4 Extreme ratios between height/width  
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((10000, 1),    #20      # 4.4 Extreme ratios between height/width 
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((32, 64),      #21      # 4.1 Divisible by 32
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((160, 96),     #22      # 4.1 Divisible by 32
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),
            pytest.param((17, 41),      #23      # 4.2 Prime numbers
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.run_in_pp]),
            pytest.param((89, 3),       #24      # 4.2 Prime numbers
                         marks=[pytest.mark.xfail(reason=FailingReasons.MICROBATCHING_UNSUPPORTED),
                                pytest.mark.slow]),

            # 3-dimensional shape, microbatch_size = 1:
            pytest.param((1, 3, 4),           marks=pytest.mark.run_in_pp),  #25     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 45, 17),         marks=pytest.mark.slow),       #26     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 1, 23),          marks=pytest.mark.slow),       #27     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 64, 1),          marks=pytest.mark.slow),       #28     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 100, 100),       marks=pytest.mark.slow),       #29     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 1000, 100),      marks=pytest.mark.slow),       #30     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 10, 1000),       marks=pytest.mark.slow),       #31     # 4.4 Extreme ratios between height/width
            pytest.param((1, 9920, 1),        marks=pytest.mark.slow),       #32     # 4.4 Extreme ratios between height/width
            pytest.param((1, 10000, 1),       marks=pytest.mark.slow),       #33     # 4.4 Extreme ratios between height/width 
            pytest.param((1, 32, 64),         marks=pytest.mark.run_in_pp),  #34     # 4.1 Divisible by 32
            pytest.param((1, 160, 96),        marks=pytest.mark.slow),       #35     # 4.1 Divisible by 32
            pytest.param((1, 17, 41),         marks=pytest.mark.slow),       #36     # 4.2 Prime numbers
            pytest.param((1, 89, 3),          marks=pytest.mark.slow),       #37     # 4.2 Prime numbers

             # 3-dimensional shape, microbatch_size > 1:
            pytest.param((2, 3, 4),           marks=pytest.mark.run_in_pp),  #38     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((11, 45, 17),        marks=pytest.mark.slow),       #39     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((11, 1, 23),         marks=pytest.mark.slow),       #40     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((11, 64, 1),         marks=pytest.mark.slow),       #41     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((100, 100, 100),     marks=pytest.mark.slow),       #42     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((10, 1000, 100),     marks=pytest.mark.slow),       #43     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((10, 10000, 1),      marks=pytest.mark.slow),       #44     # 4.4 Extreme ratios between height/width
            pytest.param((32, 32, 64),        marks=pytest.mark.slow),       #45     # 4.1 Divisible by 32
            pytest.param((64, 160, 96),       marks=pytest.mark.slow),       #46     # 4.1 Divisible by 32
            pytest.param((11, 17, 41),        marks=pytest.mark.run_in_pp),  #47     # 4.2 Prime numbers
            pytest.param((13, 89, 3),         marks=pytest.mark.slow),       #48     # 4.2 Prime numbers

            # 4-dimensional shape, microbatch_size = 1:
            pytest.param((1, 2, 3, 4),        marks=pytest.mark.run_in_pp),  #49     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 11, 45, 17),     marks=pytest.mark.slow),       #50     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 11, 1, 23),      marks=pytest.mark.slow),       #51     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 11, 64, 1),      marks=pytest.mark.slow),       #52     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 100, 100, 100),  marks=pytest.mark.slow),       #53     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 10, 1000, 100),  marks=pytest.mark.slow),       #54     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 1, 10, 1000),    marks=pytest.mark.slow),       #55     # 4.4 Extreme ratios between height/width
            pytest.param((1, 1, 9920, 1),     marks=pytest.mark.slow),       #56     # 4.4 Extreme ratios between height/width
            pytest.param((1, 10, 10000, 1),   marks=pytest.mark.slow),       #57     # 4.4 Extreme ratios between height/width
            pytest.param((1, 32, 32, 64),     marks=pytest.mark.run_in_pp),  #58     # 4.1 Divisible by 32
            pytest.param((1, 64, 160, 96),    marks=pytest.mark.slow),       #59     # 4.1 Divisible by 32
            pytest.param((1, 11, 17, 41),     marks=pytest.mark.slow),       #60     # 4.2 Prime numbers
            pytest.param((1, 13, 89, 3),      marks=pytest.mark.slow),       #61     # 4.2 Prime numbers

            # 4-dimensional shape, microbatch_size > 1:
            pytest.param((3, 11, 45, 17),     marks=pytest.mark.run_in_pp),  #62     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((2, 2, 3, 4),        marks=pytest.mark.slow),       #63     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((4, 11, 1, 23),      marks=pytest.mark.slow),       #64     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((5, 11, 64, 1),      marks=pytest.mark.slow),       #65     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((6, 100, 100, 100),  marks=pytest.mark.slow),       #66     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((7, 10, 1000, 100),  marks=pytest.mark.slow),       #67     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((8, 1, 10, 1000),    marks=pytest.mark.slow),       #68     # 4.4 Extreme ratios between height/width
            pytest.param((9, 1, 9920, 1),     marks=pytest.mark.slow),       #69     # 4.4 Extreme ratios between height/width
            pytest.param((10, 10, 10000, 1),  marks=pytest.mark.slow),       #70     # 4.4 Extreme ratios between height/width
            pytest.param((11, 32, 32, 64),    marks=pytest.mark.slow),       #71     # 4.1 Divisible by 32
            pytest.param((12, 64, 160, 96),                                  #72     # 4.1 Divisible by 32
                                marks=pytest.mark.skip(reason="RuntimeError: Fatal Python error: Segmentation fault")),
            pytest.param((13, 11, 17, 41),    marks=pytest.mark.run_in_pp),  #73     # 4.2 Prime numbers
            pytest.param((14, 13, 89, 3),     marks=pytest.mark.slow),       #74     # 4.2 Prime numbers
    ]

@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("input_shape", get_input_shapes())
def test_pytorch_eltwise_binary_ops_per_test_plan(
    input_operator,
    model_type,
    input_shape,
    test_device,
    dev_data_format=None, 
    input_math_fidelity=None
):
    
    input_source_flag = None
    if model_type == ModelFromDramQueue:
        input_source_flag = InputSourceFlags.FROM_DRAM

    kwargs = {}
    if input_operator in ["add", "sub", "substract"] and kwargs_limiter.is_allowed():
        kwargs['alpha'] = random.uniform(0.5, 1000)
    elif input_operator in ["div", "divide"]:
        rounding_modes = ['trunc', 'floor', None]
        kwargs['rounding_mode'] = rounding_modes[random.randint(0, 2)]


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
            if key == "target_device":
                continue
            assert input_operator not in key


rng_limiter = random.Random(0)
kwargs_limiter = RateLimiter(rng_limiter, 100, 50)


def get_not_implemented_pytorch_binary_ops():
    return [
        "atan2",                    #00           - NotImplementedError: The following operators are not implemented: ['aten::atan2']
        "arctan2",                  #01           - NotImplementedError: The following operators are not implemented: ['aten::atan2']
        "bitwise_and",              #02           - RuntimeError: "bitwise_and_cpu" not implemented for 'Float'
        "bitwise_or",               #03           - RuntimeError: "bitwise_or_cpu" not implemented for 'Float'
        "bitwise_xor",              #04           - RuntimeError: "bitwise_xor_cpu" not implemented for 'Float'
        "bitwise_left_shift",       #05           - RuntimeError: "lshift_cpu" not implemented for 'Float'
        "bitwise_right_shift",      #06           - RuntimeError: "rshift_cpu" not implemented for 'Float'
        "floor_divide",             #07           - AssertionError: Encountered unsupported op types. Check error logs for more details
        "fmod",                     #08           - AssertionError: Encountered unsupported op types. Check error logs for more details
        "logaddexp",                #09           - NotImplementedError: The following operators are not implemented: ['aten::logaddexp']
        "logaddexp2",               #10           - NotImplementedError: The following operators are not implemented: ['aten::logaddexp2']
        "nextafter",                #11           - NotImplementedError: The following operators are not implemented: ['aten::nextafter']
        "remainder",                #12           - AssertionError: Encountered unsupported op types. Check error logs for more details
        "fmax",                     #13           - NotImplementedError: The following operators are not implemented: ['aten::fmax']
        "fmin",                     #14           - NotImplementedError: The following operators are not implemented: ['aten::fmin']
    ]

input_shapes=[
    (1, 2, 3, 4),
]


@pytest.mark.parametrize("input_operator", get_not_implemented_pytorch_binary_ops())
@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)
def test_not_implemented_pytorch_eltwise_binary_ops_per_test_plan(
    input_operator,
    model_type,
    input_shape,
    test_device,
    dev_data_format=None, 
    input_math_fidelity=None
):

    verify(
        test_device=test_device,
        model_type=model_type,
        input_operator=input_operator,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=input_math_fidelity,
    )
