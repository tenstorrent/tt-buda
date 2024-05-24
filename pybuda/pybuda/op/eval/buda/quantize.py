# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentError
from random import random
import numpy as np
import os
import torch

from loguru import logger

import pybuda._C.balancer as balancer
from pybuda._C import DataFormat, MathFidelity
from pybuda._C.backend_api import get_op_model_execution_cycles
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile, align_up
from pybuda._C.graph import UBlockOrder

from ..common import op_model_to_desc, get_compiler_cached_cycles
from pybuda.op.eval.pybuda.quantize import STRING_TO_LOWER_LIMIT, STRING_TO_UPPER_LIMIT, STRING_TO_TORCH_DTYPE

def eval(type, attr, ops):
    if type == "quantization":
        zero_point, axis, out_dtype = attr
        input_float = ops[0].float()
        scale = ops[1].float()
        output_float = torch.clamp(
            torch.round(input_float * scale) + zero_point,
            STRING_TO_LOWER_LIMIT[out_dtype],
            STRING_TO_UPPER_LIMIT[out_dtype],)
        
        return output_float.to(STRING_TO_TORCH_DTYPE[out_dtype])
        
    elif type == "dequantization":
        zero_point, axis = attr
        input_float = ops[0].float()
        scale = ops[1].float()
        output_float = (input_float - zero_point) * scale
        return output_float

    elif type == "requantization":
        zp = attr[0]
        input_float = ops[0].float()
        scale = ops[1]
        out_dtype = "torch.int8"
        output_float = torch.round(input_float * scale + zp)
        output_float = torch.clamp(
            output_float,
            STRING_TO_LOWER_LIMIT[out_dtype],
            STRING_TO_UPPER_LIMIT[out_dtype],)

        return output_float.to(STRING_TO_TORCH_DTYPE[out_dtype])
    
def shape(type, attr, ops, tile_height, tile_width):
    broadcast = []

    if type == "quantization":
        op1 = list(ops[1])
        while len(op1) < len(ops[0]):
            op1 = [1] + op1
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != op1[dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))

    if type == "requantization":
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != ops[1][dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))

    if type == "dequantization":
        op1 = list(ops[1])
        while len(op1) < len(ops[0]):
            op1 = [1] + op1
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != op1[dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))

    return ops[0], broadcast

def parallelization(type, attr, op_shape):

    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)

def input_ublock_order(type, attr, num_operands):
    return None

def execution_cycles(type, arch_name, op_model) -> int:
    op_model_desc = op_model_to_desc(type, arch_name, op_model)

    # for dequant and requant input0 format is important,
    # output format is always Int8 for requant and Float32 for dequant;
    # this is a workaround until we expand the API to accept all data formats for an op
    if (op_model_desc.type == "dequantization" or op_model_desc.type == "requantization"):
        op_model_desc.data_format = op_model.input_buffers[0].data_format

    compiler_cache_cycles = get_compiler_cached_cycles(op_model_desc)
    if compiler_cache_cycles is not None:
        return compiler_cache_cycles

    cycle_count = get_op_model_execution_cycles(op_model_desc)
    return cycle_count
