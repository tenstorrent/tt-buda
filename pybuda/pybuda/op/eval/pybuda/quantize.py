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
from .reciprocal import Reciprocal

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
        input_int8 = ops[0].float()
        scale = ops[1].float()

        if axis < 0:
            axis = len(input_int8.shape) + axis
        left_ndim = axis
        right_ndim = len(input_int8.shape) - axis - 1
        if len(scale.shape) == 1:
            target_shape = [1] * left_ndim + list(scale.shape) + [1] * right_ndim

        if target_shape[axis] != input_int8.shape[axis]:
            assert target_shape[axis] == 1
            scale = torch.broadcast_to(scale, target_shape)
        scale = torch.reshape(scale, target_shape)

        output_float = (input_int8 - zero_point) * scale
        return output_float

    elif type == "buda_dequantize":
        zero_point, axis = attr
        input_int8 = ops[0].float()
        scale = ops[1].float()
        output_float = (input_int8 - zero_point) * scale
        return output_float

    elif type == "requantize":
        out_zp, inp_zp, axis, rounding, out_dtype = attr
        input_int32 = ops[0]
        inp_scale, out_scale, = ops[1], ops[2]
        output_scale = inp_scale / out_scale

        if axis < 0:
            axis = len(input_int32.shape) + axis
        left_ndim = axis
        right_ndim = len(input_int32.shape) - axis - 1
        if len(output_scale.shape) == 1:
            target_shape = [1] * left_ndim + list(output_scale.shape) + [1] * right_ndim

        if target_shape[axis] != input_int32.shape[axis]:
            assert target_shape[axis] == 1
            output_scale = torch.broadcast_to(output_scale, target_shape)
        output_scale = torch.reshape(output_scale, target_shape)


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
    op0 = ops[0]
    op1 = ops[1]

    if type == "quantize" or type == "buda_quantize":
        axis = attr[1]
        if axis < 0:
            axis = len(ops[0]) + axis
        left_ndim = axis
        right_ndim = len(ops[0]) - axis - 1
        if len(op1) == 1:
            op1 = [1] * left_ndim + list(ops[1]) + [1] * right_ndim
        elif len(op1) < len(op0):
            while len(op1) < len(op0):
                op1 = [1] + op1
        assert len(op1) == len(op0), "Scale and input must have same dimension"
        for dim in range(1, len(op0)):
            if op0[dim] != op1[dim]:
                broadcast.append((1, dim - len(op0), op0[dim]))

    if type == "buda_requantize" or type == "buda_dequantize":
        for dim in range(1, len(ops[0])):
            if ops[0][dim] != ops[1][dim]:
                broadcast.append((1, dim - len(ops[0]), ops[0][dim]))
    return ops[0], broadcast


def lower(type, attr, lc, ops, outputs):
    if type == "buda_quantize":
        lc.op("quantization", ops, attr, {"zero_point": attr[0]}, "", TILE_DIM, TILE_DIM) # straight 1-1 for all other binaries
    elif type == "buda_dequantize":
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
        scale = dc.op(Reciprocal.create(), [scale], output_df=scale.output_df)
        out = dc.op("buda_quantize", [inputs[0], scale], attrs=attr, output_df=buda_dtype)
        dc.fuse(out)
        return

    if type == "requantize":
        act, inp_scale, out_scale = inputs
        out_zp,inp_zp, axis, rounding, out_dtype = attr
        inp_scale_shape = inp_scale.shape.as_list()

        if axis < 0:
            axis = len(act.shape) + axis
        left_ndim = axis
        right_ndim = len(act.shape) - axis - 1
        if len(inp_scale_shape) == 1:
            # Match ndim with actiavtion
            for i in range(0, left_ndim):
                inp_scale = dc.op("unsqueeze", [inp_scale], attrs=(0, len(inp_scale_shape)), output_df=inp_scale.output_df)
                inp_scale_shape = [1] + inp_scale_shape
            for i in range(0, right_ndim):
                inp_scale = dc.op("unsqueeze", [inp_scale], attrs=(len(inp_scale_shape), len(inp_scale_shape)), output_df=inp_scale.output_df)
                inp_scale_shape = inp_scale_shape + [1]

        out_scale_shape = out_scale.shape.as_list()
        if len(out_scale_shape) == 1:
            # Match ndim with actiavtion
            for i in range(0, left_ndim):
                out_scale = dc.op("unsqueeze", [out_scale], attrs=(0, len(out_scale_shape)), output_df=out_scale.output_df)
                out_scale_shape = [1] + out_scale_shape
            for i in range(0, right_ndim):
                out_scale = dc.op("unsqueeze", [out_scale], attrs=(len(out_scale_shape), len(out_scale_shape)), output_df=out_scale.output_df)
                out_scale_shape = out_scale_shape + [1]


        if out_scale_shape[axis] != act.shape[axis]:
            assert out_scale_shape[axis] == 1
            out_scale = dc.op("broadcast", [out_scale], attrs=(axis - len(out_scale_shape), act.shape[axis]),output_df=out_scale.output_df)
            out_scale_shape[axis] = act.shape[axis]

        recip_out_scale = dc.op(Reciprocal.create(), [out_scale],output_df=out_scale.output_df,)    
        new_scale = dc.op("multiply", [inp_scale, recip_out_scale],output_df=out_scale.output_df,)

        torch_dtype = STRING_TO_TORCH_DTYPE[out_dtype]
        buda_dtype = pytorch_dtype_to_buda_dataformat(torch_dtype)
        out = dc.op("buda_requantize", [act, new_scale], attrs=(out_zp, axis, rounding, out_dtype),output_df=buda_dtype)
        dc.fuse(out)
        return

    if type == "dequantize":
        zero_point, axis = attr
        act = inputs[0]
        scale = inputs[1]
        if axis < 0:
            axis = len(act.shape) + axis
        left_ndim = axis
        right_ndim = len(act.shape) - axis - 1

        scale_shape = scale.shape.as_list()
        if len(scale_shape) == 1:
            # Match ndim with actiavtion
            for i in range(0, left_ndim):
                scale = dc.op("unsqueeze", [scale], attrs=(0, len(scale_shape)), output_df=scale.output_df)
                scale_shape = [1] + scale_shape
            for i in range(0, right_ndim):
                scale = dc.op("unsqueeze", [scale], attrs=(len(scale_shape), len(scale_shape)), output_df=scale.output_df)
                scale_shape = scale_shape + [1]


        out = dc.op("buda_dequantize", [act, scale], attrs=attr,)
        dc.fuse(out)
        return
