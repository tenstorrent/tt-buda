# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of concatenate operator
#
#
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
# (+)  3.3 Scalar
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
# (+)  5.1 Output DF
# (+)  5.2 Intermediate DF
# (+)  5.3 Accumulation DF
# (+)  5.4 Operand DFs
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example


import pytest

import pybuda
import torch

from typing import List, Dict
from loguru import logger

from pybuda import PyBudaModule
from pybuda.op_repo import TensorShape
from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import NetlistValidation
from test.operators.utils import FailingReasons
from test.conftest import TestDevice


def verify(
    test_device: TestDevice,
    model: PyBudaModule,
    input_shape: TensorShape,
    number_of_operands: int,
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: pybuda.DataFormat = None,
    math_fidelity: pybuda.MathFidelity = None,
):
    '''Common verification function for all tests'''

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


# Concatenate operator doesn't work for axis is equal to 0.
# For input shapes different from (1, n) or (n, m) the following error is raised:
# Error message:
#     "...
#      pybuda._C.UnsupportedHWOpsError: Splice op can only operate on dims 1, 2, or 3
#      ..."
# In case of shape = (n, m) the following error is raised:
# Error message:
#     "...
#      AssertionError: Error during inference
#      ..."
# In case of shape = (1, n) the test passes!
axises = [0]
input_shapes = [
                 (1, 3),            # shape0 - test passes.
                 (5, 3),            # shape1 - test fails. Message: "AssertionError: Error during inference"
                 (1, 3, 3),         # shape2 - test fails. Message: "pybuda._C.UnsupportedHWOpsError: Splice op can only operate on dims 1, 2, or 3"
                 (2, 3, 3),         # shape3 - test fails. Message: "pybuda._C.UnsupportedHWOpsError: Splice op can only operate on dims 1, 2, or 3"
                 (1, 3, 3, 3),      # shape4 - test fails. Message: "pybuda._C.UnsupportedHWOpsError: Splice op can only operate on dims 1, 2, or 3"
                 (1, 3, 3, 3, 3)    # shape5 - test fails. Message: "pybuda._C.UnsupportedHWOpsError: Splice op can only operate on dims 1, 2, or 3"
               ]
# Concatenate operator doesn't work for axis value of 0.
@pytest.mark.xfail(reason=FailingReasons.UNSUPORTED_AXIS)
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", input_shapes)
def test_concatenate_invalid_axis(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x, y):
            output = pybuda.op.Concatenate("Concatenate0", x, y, axis=axis)
            return output

    mod = Model("test_concatenate_invalid_axis_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


# setup of axises and shapes for all tests:
axises = [-3, -2, -1, 1, 2]

def get_input_shapes(microbatch_size=1):
                                              # Here we cover interesting combinations of input shapes:
    return [
            (microbatch_size, 3, 3),         # 3.1 Full tensor (i.e. full expected shape)
            (microbatch_size, 10, 5),        # 3.1 Full tensor (i.e. full expected shape)
            (microbatch_size, 1, 15),        # 3.2 Tensor reduce on one or more dims to 1
            (microbatch_size, 50, 1),        # 3.2 Tensor reduce on one or more dims to 1
            (microbatch_size, 100, 100),     # 4.3 Very large (thousands, 10s of thousands)
            (microbatch_size, 100, 1000),    # 4.3 Very large (thousands, 10s of thousands)
            (microbatch_size, 1, 4991),      # 4.4 Extreme ratios between height/width        - FAILING FOR 4992 and axis=[-1, 2]
            (microbatch_size, 8191, 1),      # 4.4 Extreme ratios between height/width        - FAILING FOR 8192 and axis=[-1, 2]
            (microbatch_size, 32, 32),       # 4.1 Divisible by 32
            (microbatch_size, 96, 96),       # 4.1 Divisible by 32
            (microbatch_size, 13, 97),       # 4.2 Prime numbers
            ]


#   2.1 From another op
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
def test_concatenate_inputs_from_another_operand(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x, y):
            # we use Add and Subtract operators to create two operands which are inputs for the Concatenate operator
            xx = pybuda.op.Add("Add0", x, y)
            yy = pybuda.op.Subtract("Subtract0", x, y)
            output = pybuda.op.Concatenate("Concatenate0", xx, yy, axis=axis)
            return output
        
    mod = Model("test_concatenate_inputs_from_another_operand_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


#   2.2 From tm edge
#    - Combination: operator -> tm -> input
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
def test_concatenate_inputs_from_tm_edge1(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x, y):
            v1 = pybuda.op.Add("Add0", x, y)
            v2 = pybuda.op.tm.Transpose("Transpose0", v1, -1, -2)
            v3 = pybuda.op.Concatenate("Concatenate0", v2, v2, axis=axis)
            return v3
        
    mod = Model("test_concatenate_inputs_from_tm_edge1_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


#   2.2 From tm edge
#    - tm -> input
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
def test_concatenate_inputs_from_tm_edge2(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x, y):
            v1 = pybuda.op.tm.Transpose("Transpose0", x, -1, -2)
            v2 = pybuda.op.tm.Transpose("Transpose1", y, -1, -2)
            v3 = pybuda.op.Concatenate("Concatenate0", v1, v2, axis=axis)
            return v3
        
    mod = Model("test_concatenate_inputs_from_tm_edge2_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


#   2.3 From DRAM queue
#    - input_queue flag = false
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
def test_concatenate_inputs_from_dram_queue(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x, y):
            return pybuda.op.Concatenate("Concatenate0", x, y, axis=axis)
        
    mod = Model("test_concatenate_inputs_from_dram_queue_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=2,
        input_source_flag=InputSourceFlags.FROM_DRAM,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

    netlist = NetlistValidation()
    assert netlist.get_value("/queues/x/loc") == 'dram'
    assert netlist.get_value("/queues/y/loc") == 'dram'



#   2.4 From DRAM, but prologued (constant)
#    - Constants must be small enough to fit into L1
#    - Input are not prologued for microbatch size = 1
# FAILING FOR axis=[-3], but pass for others
@pytest.mark.parametrize("axis", [pytest.param(-3, marks=pytest.mark.xfail(reason=FailingReasons.UNSUPORTED_AXIS)),
                                  pytest.param(-2),
                                  pytest.param(-1),
                                  pytest.param(1),
                                  pytest.param(2)
                                  ])
@pytest.mark.parametrize("input_shape, input_source_flag, should_prolog", [
    pytest.param((2, 3, 3),      InputSourceFlags.FROM_DRAM_NOT_PROLOGUED,            False),                                                              # 3.1 Full tensor (i.e. full expected shape)    - FAILING FOR axis=[-3]
    pytest.param((2, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUED,                True),                                                               # 3.1 Full tensor (i.e. full expected shape)    - FAILING FOR axis=[-3]
    pytest.param((2, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 3.1 Full tensor (i.e. full expected shape)    - FAILING FOR axis=[-3]
    pytest.param((1, 3, 3),      InputSourceFlags.FROM_DRAM_NOT_PROLOGUED,            False),                                                              # 3.1 Full tensor (i.e. full expected shape)    - PASS
    pytest.param((1, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUED,                True),                                                               # 3.1 Full tensor (i.e. full expected shape)    - PASS
    pytest.param((1, 3, 3),      InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 3.1 Full tensor (i.e. full expected shape)    - PASS - but not according to documentation!
    pytest.param((2, 10, 5),     InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 3.1 Full tensor (i.e. full expected shape)    - FAILING FOR axis=[-3]
    pytest.param((2, 1, 15),     InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 3.2 Tensor reduce on one or more dims to 1    - FAILING FOR axis=[-3]
    pytest.param((2, 50, 1),     InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 3.2 Tensor reduce on one or more dims to 1    - FAILING FOR axis=[-3]
    pytest.param((2, 100, 100),  InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 4.3 Very large (thousands, 10s of thousands)  - FAILING FOR axis=[-3]
    # FAILING FOR axis=[-3, -1, 2]
    pytest.param((2, 100, 1000), InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False, marks=pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),  # 4.3 Very large (thousands, 10s of thousands)
    # FAILING FOR for all axises
    pytest.param((2, 1, 4991),   InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False, marks=pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),  # 4.4 Extreme ratios between height/width
    # FAILING FOR axis=[-3, -1, 2]
    pytest.param((2, 1, 10000),  InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False, marks=pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),  # 4.4 Extreme ratios between height/width
    # FAILING FOR for all axises
    pytest.param((2, 8191, 1),   InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False, marks=pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),  # 4.4 Extreme ratios between height/width
    # FAILING FOR axis=[-3, -1, 2]
    pytest.param((2, 10000, 1),  InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, False, marks=pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),  # 4.4 Extreme ratios between height/width
    pytest.param((2, 32, 32),    InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 4.1 Divisible by 32                           - FAILING FOR axis=[-3]
    pytest.param((2, 96, 96),    InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 4.1 Divisible by 32                           - FAILING FOR axis=[-3]
    pytest.param((2, 13, 97),    InputSourceFlags.FROM_DRAM_PROLOGUE_MICROBATCH_SIZE, True),                                                               # 4.2 Prime numbers                             - FAILING FOR axis=[-3]
])
def test_concatenate_inputs_from_dram_prologued(test_device, axis, input_shape, input_source_flag, should_prolog, dev_data_format=None, math_fidelity=None):
    
    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

            def my_rand(*shape, requires_grad=False):
                return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()

            self.shape_input = ShapeUtils.reduce_microbatch_size(input_shape)

            self.add_constant("c")
            self.set_constant("c", pybuda.Tensor.create_from_torch(my_rand(*self.shape_input), constant=True))


        def forward(self, x):
            return pybuda.op.Concatenate("Concatenate0", self.get_constant("c"), x, axis=axis)
        
    mod = Model("test_concatenate_inputs_from_dram_prologued_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=1,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

    netlist = NetlistValidation()
    d = netlist.get_value("/programs/0/run_fwd_0/4/execute/queue_settings/input_0_Concatenate0")
    if should_prolog:
        assert d['prologue']
    else:
        assert not d['prologue']


#   2.5 Const Inputs (const eval pass)
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
def test_concatenate_inputs_from_constants(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):
     
    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

            def my_rand(*shape, requires_grad=False):
                return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()

            self.shape_input = ShapeUtils.reduce_microbatch_size(input_shape)

            self.add_constant("c1")
            self.set_constant("c1", pybuda.Tensor.create_from_torch(my_rand(*self.shape_input), constant=True))

            self.add_constant("c2")
            self.set_constant("c2", pybuda.Tensor.create_from_torch(my_rand(*self.shape_input), constant=True))

        def forward(self, x, y):
            v1 = pybuda.op.Concatenate("Concatenate0", self.get_constant("c1"), self.get_constant("c2"), axis=axis)
            # v2 and v3 consume inputs
            v2 = pybuda.op.Add("Add0", x, y)
            v3 = pybuda.op.Add("Add1", v1, v2)
            return v3

    mod = Model("test_concatenate_inputs_from_constants_model")

    if axis % 3 == 0:
        # TODO: check - for axis = 0 concatenate doesn't change shape, maybe this is incorrect
        i_shape = input_shape
    else:
        i_shape = list(input_shape)
        i_shape[axis] = 2 * i_shape[axis]
        i_shape = tuple(i_shape)

    verify(
        test_device=test_device,
        model=mod,
        input_shape=i_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

    # Here we check there is no key with "Concatenate" in the netlist in graphs section
    netlist = NetlistValidation()
    d = netlist.get_value("/graphs/fwd_0_0_temporal_epoch_0")
    for key in d.keys():
        assert "Concatenate" not in key


#   2.6 From host - case of two tensors as input
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
def test_concatenate_inputs_from_host_2(test_device, axis, input_shape, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x, y):
            return pybuda.op.Concatenate("Concatenate0", x, y, axis=axis)
        
    mod = Model("test_concatenate_inputs_from_host_2_model")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

number_of_operands = [
                       pytest.param(3),   # all passes.
                       pytest.param(4, marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),   # fails only for GOLDEN_WORMHOLE_BO=1
                       pytest.param(7, marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),   # fails in any case
                            # Error message:
                            # ...
                            # [Golden-7-input_shape6--1] - RuntimeError: 1 Nodes have no valid grids, exiting
                            # [Golden-7-input_shape6-2] - RuntimeError: 1 Nodes have no valid grids, exiting
                            # [Golden-7-input_shape7--2] - RuntimeError: 1 Nodes have no valid grids, exiting
                            # [Golden-7-input_shape7-1] - RuntimeError: 1 Nodes have no valid grids, exiting
                            # ...
                       pytest.param(15, marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),  # fails in any case
                            # Error message:
                            # ...
                            # [Golden-15-input_shape6--1] - RuntimeError: TT_ASSERT @ pybuda/csrc/balancer/balancer_utils.cpp:238: shape.ct % factor == 0
                            # [Golden-15-input_shape6-2] - RuntimeError: TT_ASSERT @ pybuda/csrc/balancer/balancer_utils.cpp:238: shape.ct % factor == 0
                            # [Golden-15-input_shape7--2] - RuntimeError: 2 Nodes have no valid grids, exiting
                            # [Golden-15-input_shape7-1] - RuntimeError: 2 Nodes have no valid grids, exiting
                            # ...
                     ]

#   2.6 From host - case of multiple number of tensors as input
@pytest.mark.parametrize("axis", axises)
@pytest.mark.parametrize("input_shape", get_input_shapes(microbatch_size=1))
@pytest.mark.parametrize("number_of_operands", number_of_operands)
def test_concatenate_inputs_from_host_multiple_operands(test_device, axis, input_shape, number_of_operands, dev_data_format=None, math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, *x):
            return pybuda.op.Concatenate("Concatenate0", *x, axis=axis)
        
    mod = Model("test_concatenate_inputs_from_host_multiple_operands")

    verify(
        test_device=test_device,
        model=mod,
        input_shape=input_shape,
        number_of_operands=number_of_operands,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


# Operand Data Format and Math Fidelity

# First, we will test only by fixing one axis and one input shape.
axis = 1
def get_single_shape(microbatch_size=1):
    return (microbatch_size, 3, 3)        # Full tensor, small size

# We will not test all combinations of Data Format and Math Fidelity
# because it would be too much tests. 
#   1. First we will choose Data Format to be Float16_b and test all Math Fidelity values
#   2. Then we will set Math Fidelity to HiFi4 and test all Data Formats. 

### 1. ####################################################################################

#   5.4 Operand DFs
dev_data_formats=[ 
    pybuda.DataFormat.Float16_b,
]

#  6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
compiler_math_fidelity = [
                            pybuda.MathFidelity.LoFi,
                            pybuda.MathFidelity.HiFi2,
                            pybuda.MathFidelity.HiFi3,
                            pybuda.MathFidelity.HiFi4,
                         ]


@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_concatenate_mf_inputs_from_another_operand(test_device, dev_data_format, math_fidelity):
    test_concatenate_inputs_from_another_operand(test_device, axis, get_single_shape(), dev_data_format, math_fidelity)


### 2. ####################################################################################

#   5.4 Operand DFs
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

#  6. Math fidelity
compiler_math_fidelity = [
    pybuda.MathFidelity.HiFi4,
]


@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_concatenate_df_inputs_from_another_operand(test_device, dev_data_format, math_fidelity):
    test_concatenate_inputs_from_another_operand(test_device, axis, get_single_shape(), dev_data_format, math_fidelity)
