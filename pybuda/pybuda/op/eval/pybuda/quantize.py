# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import ast
import os
import math
import torch.nn.functional as F
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile
import numpy as np
from ..common import to_torch_operands
from pybuda.tensor import pytorch_dtype_to_buda_dataformat

STRING_TO_TORCH_DTYPE = {
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int32": torch.int32,
    "torch.float32": torch.float32,
}

STRING_TO_LOWER_LIMIT = {
    "torch.int8": -128,
    "torch.uint8": 0,
    "torch.int32": -2147483648,
    "torch.float32": -3.4028234663852886e+38,
}

STRING_TO_UPPER_LIMIT = {
    "torch.int8": 127,
    "torch.uint8": 255,
    "torch.int32": 2147483647,
    "torch.float32": 3.4028234663852886e+38,
}

def eval(type, attr, ops):
    if type == "quantize":
        zero_point, axis, out_dtype = attr
        input_float = ops[0].float()
        scale = ops[1].float()
        output_float = torch.clamp(
            torch.round(input_float / scale) + zero_point,
            STRING_TO_LOWER_LIMIT[out_dtype],
            STRING_TO_UPPER_LIMIT[out_dtype],)
        return output_float.to(STRING_TO_TORCH_DTYPE[out_dtype])

    elif type == "buda_quantize":
        zero_point, axis, out_dtype = attr
        input_float = ops[0].float()
        scale = ops[1].float()
        output_float = torch.clamp(
            torch.round(input_float * scale) + zero_point,
            STRING_TO_LOWER_LIMIT[out_dtype],
            STRING_TO_UPPER_LIMIT[out_dtype],)
        return output_float.to(STRING_TO_TORCH_DTYPE[out_dtype])

    elif type == "dequantize":
        zero_point, axis = attr
        input_int8 = ops[0]
        scale = ops[1]
        output_float = (input_int8 - zero_point) * scale
        return output_float

    elif type == "requantize":
        out_zp, inp_zp, axis, rounding, out_dtype = attr
        input_int32 = ops[0]
        inp_scale, out_scale, = ops[1], ops[2]
        output_scale = inp_scale / out_scale
        while len(output_scale.shape) != len(input_int32.shape):
            output_scale = output_scale.unsqueeze(-1)

        assert inp_zp == 0, "Only support input zero point of 0"
        output_float = torch.round(output_scale * (input_int32 - inp_zp) + out_zp)
        output_float = torch.clamp(
            output_float,
            STRING_TO_LOWER_LIMIT[out_dtype],
            STRING_TO_UPPER_LIMIT[out_dtype],)

        return output_float.to(STRING_TO_TORCH_DTYPE[out_dtype])

    elif type == "buda_requantize":
        zp, axis, rounding, out_dtype = attr
        input_int32 = ops[0].float()
        scale = ops[1].float()
        output_float = torch.round(input_int32 * scale + zp)
        output_float = torch.clamp(
            output_float,
            STRING_TO_LOWER_LIMIT[out_dtype],
            STRING_TO_UPPER_LIMIT[out_dtype],)

        return output_float.to(STRING_TO_TORCH_DTYPE[out_dtype])

def shape(type, attr, ops):
    broadcast = []

    if type == "quantize" or type == "buda_quantize":
        op1 = list(ops[1])
        while len(op1) < len(ops[0]):
            op1 = [1] + op1
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != op1[dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))

    if type == "buda_requantize":
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != ops[1][dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))

    if type == "dequantize":
        op1 = list(ops[1])
        while len(op1) < len(ops[0]):
            op1 = [1] + op1
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != op1[dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))

    return ops[0], broadcast


def lower(type, attr, lc, ops, outputs):
    if type == "buda_quantize":
        lc.op("quantization", ops, attr, {"zero_point": attr[0]}, "", TILE_DIM, TILE_DIM) # straight 1-1 for all other binaries
    elif type == "dequantize":
        lc.op("dequantization", ops, attr, {"zero_point": attr[0]}, "", TILE_DIM, TILE_DIM) # straight 1-1 for all other binaries
    elif type == "buda_requantize":
        lc.op("requantization", ops, attr, {"zero_point": attr[0]}, "", TILE_DIM, TILE_DIM)

    else:
        raise RuntimeError(f"Unknown quantize type {type}")
    
def backward(type, attr, ac, operand, inputs, output, grad):
    assert False, "Quantize does not support backward"

def decompose(type, attr, dc, inputs):
    if type == "quantize":
        zero_point, axis, out_dtype = attr
        torch_dtype = STRING_TO_TORCH_DTYPE[out_dtype]
        buda_dtype = pytorch_dtype_to_buda_dataformat(torch_dtype)
        scale = inputs[1]
        scale = dc.op("reciprocal", [scale], output_df=scale.output_df)
        out = dc.op("buda_quantize", [inputs[0], scale], attrs=attr, output_df=buda_dtype)
        dc.fuse(out)
        return

    if type == "requantize":
        act, inp_scale, out_scale = inputs
        out_zp,inp_zp, axis, rounding, out_dtype = attr
        inp_scale_shape = inp_scale.shape.as_list()
        if len(inp_scale_shape) == 1:
            # populate batch dim
            inp_scale = dc.op("unsqueeze", [inp_scale], attrs=(0, len(inp_scale_shape)), output_df=inp_scale.output_df)
            inp_scale_shape = [1] + inp_scale_shape
        
        while len(inp_scale_shape) < len(act.shape.as_list()):
            inp_scale = dc.op("unsqueeze", [inp_scale], attrs=(len(inp_scale_shape), len(inp_scale_shape)), output_df=inp_scale.output_df)
            inp_scale_shape = inp_scale_shape + [1]


        out_scale_shape = out_scale.shape.as_list()
        if len(out_scale_shape) == 1:
            # populate batch dim
            out_scale = dc.op("unsqueeze", [out_scale], attrs=(0, len(out_scale_shape)), output_df=out_scale.output_df)
            out_scale_shape = [1] + out_scale_shape
        
        while len(out_scale_shape) < len(act.shape.as_list()):
            out_scale = dc.op("unsqueeze", [out_scale], attrs=(len(out_scale_shape), len(out_scale_shape)), output_df=out_scale.output_df)
            out_scale_shape = out_scale_shape + [1]


        for i, (left, right) in enumerate(zip(inp_scale_shape, out_scale_shape)):
            if i == 0:
                continue

            if left != right:
                out_scale = dc.op("broadcast", [out_scale], attrs=(i - len(out_scale_shape), left),output_df=out_scale.output_df)            

        recip_out_scale = dc.op("reciprocal", [out_scale],output_df=out_scale.output_df,)
        new_scale = dc.op("multiply", [inp_scale, recip_out_scale],output_df=out_scale.output_df,)

        out = dc.op("buda_requantize", [act, new_scale], attrs=(out_zp, axis, rounding, out_dtype),)
        dc.fuse(out)
        return

