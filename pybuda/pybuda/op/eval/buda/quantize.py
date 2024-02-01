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

from ..common import to_torch_operands, math_fidelity_to_multiplier, data_format_to_int, op_model_to_desc
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
        input_float = ops[0]
        scale = ops[1]
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

def parallelization(type, attr, op_shape, fracture_factor):

    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)

def input_ublock_order(type, attr, num_operands):
    return None

def execution_cycles(type, arch_name, op_model) -> int:
    b = op_model.output_buffers[0].block_shape
    cycles_per_tile = 32 * 20
    return b.mblock_m * b.mblock_n * b.ublock.rt * b.ublock.ct * b.t * cycles_per_tile
