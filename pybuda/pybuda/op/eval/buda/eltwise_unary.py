# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger

import pybuda._C.balancer as balancer
from pybuda.pybudaglobal import TILE_DIM
import torch
import torch.nn.functional
from pybuda.utils import align_up_tile
from pybuda._C import DataFormat, MathFidelity
from pybuda._C.graph import UBlockOrder, Shape
from pybuda._C.backend_api import get_op_model_param
from .tm import eval as tm_eval
from pybuda.tensor import pad_pytorch_tensor_to_buda
from pybuda._C.backend_api import get_op_model_execution_cycles

from ..common import to_torch_operands, op_model_to_desc, get_compiler_cached_cycles

M_2_SQRTPI  = 1.12837916709551257390	# 2/sqrt(pi) 
M_SQRT2     = 1.41421356237309504880	# sqrt(2) 
M_SQRT1_2   = 0.7071067811865476

# Reference implementation is at pytorch/aten/src/ATen/native/cpu/Activation.cpp
# https://github.com/pytorch/pytorch/blob/4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8/aten/src/ATen/native/cpu/Activation.cpp
def gelu_derivative(x, approximate):
    if approximate == "none":
        cdf = 0.5 * (1 + torch.erf(x * M_SQRT1_2))
        pdf = 0.5 * M_SQRT1_2 * M_2_SQRTPI * torch.exp(x * x * -0.5)
        return cdf + x * pdf
    elif approximate == "tanh":
        intermediate_0 = 0.5 * (1 + torch.tanh((M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * torch.pow(x, 3))))
        intermediate_1 = x * torch.exp(-0.5 * x * x) * (0.5 * M_2_SQRTPI / M_SQRT2)
        return intermediate_0 + intermediate_1
    else:
        raise RuntimeError(f"Gelu does not support {approximate} approximation mode.")

def gelu_forward(x, approximate):
    if approximate == "none":
        return torch.nn.functional.gelu(x)
    elif approximate == "tanh":
        import math
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    else:
         raise RuntimeError(f"Gelu does not support {approximate} approximation mode.")


def eval(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    # assert (type != "ethernet_datacopy" or (len(attr) == 1 or len(attr) == 2)), f"Ethernet datacopy must only have 1 or 2 attributes. Attrs = {attr}" tenstorrent/pybuda#1085
    
    t_ops = to_torch_operands(*ops)

    # Some ops don't support non-fp32 in pytorch
    original_types = [o.dtype for o in t_ops]
    if original_types[0] != torch.float32:
        if type in ["gelu", "gelu_derivative"]:
            t_ops = tuple(t.type(torch.float32) for t in t_ops)

    def eval_reduce(i):
        assert len(attr) == 3
        dim, reduce_op, z = attr
        assert reduce_op == "max"
        if dim > 1:
            ret = torch.max(i[0], dim=dim, keepdim=True)[0]
            r = ret.shape[-2]
            c = ret.shape[-1]
            pad_r = align_up_tile(r)
            pad_c = align_up_tile(c)
            ret = torch.nn.functional.pad(ret, (0, pad_c - c, 0, pad_r - r))
        else:
            assert dim == 1
            assert i[0].shape[1] % z == 0
            ret = i[0].squeeze(0).split(z)
            ret = [torch.max(s, dim=0)[0] for s in ret]
            ret = torch.stack(ret).unsqueeze(0)
        return  ret

    if type == "dropout":
        p, training, seed, r_pad, c_pad, t_stream_r, t_stream_c, is_r_major, is_z_major = attr
        ret = t_ops[0]

        assert not is_z_major, "ERR: z_major should be false. TStream Direction should be only R or C."

        # Undo Streaming
        if is_r_major:
            ret = tm_eval("hstack", [t_stream_c], [ret]) if t_stream_c > 1 else ret
            ret = tm_eval("vstack", [t_stream_r], [ret]) if t_stream_r > 1 else ret
        else:
            ret = tm_eval("vstack", [t_stream_r], [ret]) if t_stream_r > 1 else ret
            ret = tm_eval("hstack", [t_stream_c], [ret]) if t_stream_c > 1 else ret

        # Remove padding
        ret = ret.narrow(2, 0, r_pad)
        ret = ret.narrow(3, 0, c_pad)

        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        ret = torch.nn.functional.dropout(ret, p=p, training=bool(training))
        torch.set_rng_state(rng_state)

        # Re-pad if previously tilized
        ret = pad_pytorch_tensor_to_buda(ret, [])

        # Re-t-stream
        if is_r_major:
            ret = tm_eval("vslice", [t_stream_r], [ret]) if t_stream_r > 1 else ret
            ret = tm_eval("hslice", [t_stream_c], [ret]) if t_stream_c > 1 else ret
        else:
            ret = tm_eval("hslice", [t_stream_c], [ret]) if t_stream_c > 1 else ret
            ret = tm_eval("vslice", [t_stream_r], [ret]) if t_stream_r > 1 else ret

        return ret

    f = {
        "exp": lambda i: torch.exp(i[0]),
        "sqrt": lambda i: torch.sqrt(i[0]),
        "lrelu": lambda i: torch.nn.functional.leaky_relu(i[0], attr[0]),
        "gelu": lambda i : gelu_forward(i[0], approximate=attr[0]),
        "gelu_derivative": lambda i : gelu_derivative(i[0], approximate=attr[0]),
        "nop": lambda i: i[0],
        "ethernet_datacopy": lambda i: i[0],
        "buffer": lambda i: i[0],
        "reciprocal": lambda i: torch.reciprocal(i[0] + 1e-10), # add epsilon to avoid infinity
        "log": lambda i: torch.log(i[0] + 1e-10), # add epsilon to avoid nan
        "sigmoid": lambda i: torch.sigmoid(i[0]),
        "clip": lambda i: torch.clip(i[0], min=attr[0], max=attr[1]),
        "reduce": eval_reduce,
        "tanh": lambda i: torch.tanh(i[0]),
        "abs": lambda i: torch.abs(i[0]),
        "cosine": lambda i: torch.cos(i[0]),
        "sine": lambda i: torch.sin(i[0]),
        "power": lambda i: torch.pow(i[0], attr[0]),
        "tilizer": lambda i: i[0]
    }

    assert type in f, f"{type} not defined in eval map for eltwise unary ops."

    ret = f[type](t_ops)

    if ret.dtype != original_types[0]:
        ret = ret.type(original_types[0])

    return ret

def shape(op_type, attr, ops, tile_height, tile_width):
    assert len(ops) == 1, "Eltwise unary should have one input"

    if op_type == "reduce":
        assert len(attr) == 3
        dim, reduce_op, z = attr
        if dim >= 0:
            dim -= len(ops[0])
        shape = ops[0]
        if dim >= -2: # last 2 dims, column or row
            shape[dim] = align_up_tile(1)
        else:
            shape[dim] = ops[0][-3] // z
        return tuple(shape), []

    if tile_height == TILE_DIM:
        ops[0][-2] = align_up_tile(ops[0][-2])
    elif tile_height < TILE_DIM:
        ops[0][-2] = tile_height
    else:
        raise RuntimeError(f"Tile height {tile_height} is larger than max allowed TILE_DIM {TILE_DIM}")

    if op_type == "nop":
        # extend 4D -> 5D for unsqueeze NOP
        if len(attr) == 2 and attr[0] == "unsqueeze":
            if attr[1] == 4:
                ops_updated = Shape.create_buda([1] + ops[0], tile_height, tile_width)
                return ops_updated, []

    return ops[0], []


def parallelization(type, attr, op_shape):
    if type == "reduce":
        dim = attr[0]
        if dim == 1:
            return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)
        if dim == 2:
            return (1, op_shape.outputs[0].ct)
        if dim == 3:
            return (op_shape.outputs[0].rt, 1)
        return None

    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)


def input_ublock_order(type, attr, num_operands):
    if type == "reduce":
        dim = attr[0]
        if dim == 1:
            return None
        if dim == 2:
            return [UBlockOrder.C]
        if dim == 3:
            return [UBlockOrder.R]
        
    if type == "tilizer":
        return [UBlockOrder.R]
    return None


def execution_cycles(type, arch_name, op_model) -> int:
    op_model_desc = op_model_to_desc(type, arch_name, op_model)

    compiler_cache_cycles = get_compiler_cached_cycles(op_model_desc)
    if compiler_cache_cycles is not None:
        return compiler_cache_cycles

    use_legacy_path = bool(int(os.environ.get("PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY", "0")))

    # Some ops don't yet have implemented cycles, approximate cycles here
    # Additionally, always use the BBE path for `reduce` op
    if (use_legacy_path or type == "power" or type == "ethernet_datacopy") and not type == "reduce":
        tile_weight = get_op_model_param(op_model_desc, "tile_weight")
        output_shape = op_model.op_shape.outputs[0]
        num_tiles = (output_shape.z * output_shape.rt * output_shape.ct) / (op_model.grid_shape.r * op_model.grid_shape.c)
        cycle_count = tile_weight * num_tiles
        return min(int(cycle_count), 1 << 30)

    return get_op_model_execution_cycles(op_model_desc)
