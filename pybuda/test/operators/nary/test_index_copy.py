# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of stack operator



# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (-)  2.1 From another op
#       - Operator -> input
# (-)  2.2 From tm edge
#       - Combination: operator -> tm -> input
#       - tm -> input
# (-)  2.3 From DRAM queue
#       - input_queue flag = false
#       - Special case of From host? May it be triggered if the operator is not the first node of the network?
#       - Can this be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# (-)  2.4 From DRAM, but prologued (constant)
#       - Constants must be small enough to fit into L1
#       - Verification via netlists that scenario is triggered
#       - Input are not prologued for microbatch size = 1
# (-)  2.5 Const Inputs (const eval pass)
#       - Operator where all inputs are constants. Does it make difference if tensor is big > L1
#       - Verification via netlists that scenario is triggered???
# (-)  2.6 From host
#       - Input tensor as input of network -> Operator is first node in network and input_queue flag = true
#       - Can this scenario be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# 3 Operand shapes type(s):
# (-)  3.1 Full tensor (i.e. full expected shape)
#       - Is 3 dims max for all ops? Ex. Conv is 3d max
# (-)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (-)  3.3 Scalar
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (-)  4.1 Divisible by 32
# (-)  4.2 Prime numbers
# (-)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (-)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (-)  5.1 Output DF
# (-)  5.2 Intermediate DF
# (-)  5.3 Accumulation DF
# (-)  5.4 Operand DFs
# (-) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
# (-) 7. Special attributes - if applicable.. like approx_mode for Exp, for example


import pytest

import pybuda
import torch

from pybuda import PyBudaModule, VerifyConfig
from pybuda.config import _get_global_compiler_config
from pybuda.verify import TestKind, verify_module

# IndexCopy operator in PyBuda works in case of index is vector of one element
def test_index_copy_torch_and_buda_1():

    zeros_torch = torch.zeros(6, 3)

    x_torch = zeros_torch

    dim_torch = 0
    index_torch = torch.tensor([2], dtype=torch.long)
    source_torch = torch.tensor([[10, 10, 10]], dtype=torch.float)

    # print(f"\nx_torch before:\n{x_torch}")
    x_torch.index_copy_(dim_torch, index_torch, source_torch)
    # print(f"\nx_torch after:\n{x_torch}")

    # setting operands for pybuda:
    operandA = pybuda.tensor.Tensor.create_from_torch(zeros_torch)
    index = pybuda.tensor.Tensor.create_from_torch(index_torch)
    value = pybuda.tensor.Tensor.create_from_torch(source_torch)
    dim_buda = dim_torch

    result_buda = pybuda.op.IndexCopy("IndexCopy0", operandA, index, value, dim_buda)
    # print(f"\nresult_buda:\n{result_buda}")

    output_are_the_same = torch.eq(x_torch, pybuda.tensor.Tensor.to_pytorch(result_buda)).all()
    assert output_are_the_same


# Case of IndexCopy operator is not working
# In PyTorch, index can be tensor of any shape, but in PyBuda, it can be only vector of one element
@pytest.mark.xfail(reason="IndexCopy operator does not work")
def test_index_copy_torch_and_buda_2():

    zeros_torch = torch.zeros(6, 3)

    x_torch = zeros_torch

    dim_torch = 0
    index_torch = torch.tensor([0, 2], dtype=torch.long)
    source_torch = torch.tensor([[10, 10, 10], [20, 20, 20]], dtype=torch.float)

    print(f"\nx_torch before:\n{x_torch}")
    x_torch.index_copy_(dim_torch, index_torch, source_torch)
    print(f"\nx_torch after:\n{x_torch}")

    # setting operands for pybuda:
    operandA = pybuda.tensor.Tensor.create_from_torch(zeros_torch)
    index = pybuda.tensor.Tensor.create_from_torch(index_torch)
    value = pybuda.tensor.Tensor.create_from_torch(source_torch)

    result_buda = pybuda.op.IndexCopy("IndexCopy0", operandA, index, value, 0)
    print(f"\nresult_buda:\n{result_buda}")

    output_are_the_same = torch.eq(x_torch, pybuda.tensor.Tensor.to_pytorch(result_buda)).all()
    assert output_are_the_same


# IndexCopy operator in PyBuda is not working while testing it via model
# Running test on grayskull and wormhole devices but in both cases, it is failing with the same error:
# "...
#    WARNING  | Always          - Unsupported HW op: IndexCopy0 index_copy(axis: 0)
#    WARNING  | pybuda.compile:run_balancer_and_placer:949 - Found unsupported HW ops, stopping compilation early:
#    IndexCopy0 index_copy(axis: 0)
#    
#    ERROR    | pybuda.device:run_next_command:469 - Compile error: 'consteval_trace'
#  ...
# "
@pytest.mark.parametrize("input_shape", [(2, 3, 3)])
@pytest.mark.xfail(reason="IndexCopy operator does not work on any device.")
def test_index_copy_via_model(test_device, input_shape, input_params=[], math_fidelity=None):

    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

            self.add_constant("index")
            self.set_constant("index", pybuda.tensor.Tensor.create_from_torch(torch.tensor([0]), dev_data_format = pybuda.DataFormat.UInt16, constant=True))

        def forward(self, x, y):
            operandA = x
            index = self.get_constant("index")
            value = y
            dim = 0
            output = pybuda.op.IndexCopy("IndexCopy0", operandA, index, value, dim)
            return output
        
    mod = Model("test_index_copy_via_model_model")
    input_shapes = (input_shape,) + (((1,) + input_shape[1:]),)
    # print(f"\n\n\n**********************  input_shapes: {input_shapes}  ***************************\n\n\n")

    if(math_fidelity is not None):
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.default_math_fidelity = math_fidelity

    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )