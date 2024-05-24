# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List, Tuple
from pybuda.utils import align_up_tile, round_up_div, align_up

import pybuda._C.balancer as balancer
import torch
from pybuda._C import DataFormat, MathFidelity
from pybuda._C.graph import UBlockOrder
from pybuda._C.backend_api import get_op_model_execution_cycles

from ....pybudaglobal import TILE_DIM
from ..common import to_torch_operands, math_fidelity_to_multiplier, op_model_to_desc, get_compiler_cached_cycles


def eval(type, attr, ops):
    assert len(ops) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"
    
    t_ops = to_torch_operands(*ops)

    # Fix broadcast from 32 to 1
    for dim in range(2, 4):
        if t_ops[0].shape[dim] != t_ops[1].shape[dim]:
            if t_ops[0].shape[dim] == TILE_DIM:
                t_ops = (t_ops[0].narrow(dim, 0, 1), t_ops[1])
            else:
                t_ops = (t_ops[0], t_ops[1].narrow(dim, 0, 1))

    # Some ops don't support non-fp32 in pytorch
    original_types = [o.dtype for o in t_ops]
    if any(o != torch.float32 for o in original_types):
        if type in ["heaviside"]:
            t_ops = tuple(t.type(torch.float32) for t in t_ops)

    f = {
        "add": lambda i: torch.add(i[0], i[1]),
        "subtract": lambda i: torch.subtract(i[0], i[1]),
        "multiply": lambda i: torch.multiply(i[0], i[1]),
        "maximum": lambda i: torch.maximum(i[0], i[1]),
        "minimum": lambda i: torch.minimum(i[0], i[1]),
        "heaviside": lambda i: torch.heaviside(i[0], i[1]),
        "binary_vstack": lambda i: torch.stack((t_ops[0], t_ops[1]), axis=-1).flatten(-2),
        "binary_hstack": lambda i: torch.stack((t_ops[0], t_ops[1]), axis=-2).flatten(-3, -2),
    }
    assert type in f, f"{type} not defined in eval map for eltwise binary ops."

    ret = f[type](t_ops)
    if ret.dtype != original_types[0]:
        ret = ret.type(original_types[0])

    return ret

# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops, tile_height, tile_width) -> Tuple[Tuple, List]:
    assert len(ops) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"

    if type == "binary_hstack" or type == "binary_vstack":
        dim = -1 if type == "binary_vstack" else -2
        assert ops[0] == ops[1]
        output_shape = list(ops[0])
        output_shape[dim] *= 2
        return tuple(output_shape), []

    output_shape = ops[0]
    ops1_shape = ops[1]
    dim_r = len(output_shape) - 2

    if len(ops[0]) == 5 and len(ops[1]) == 4:
        assert ops[0][0] == 1, f"Eltwise binary ops must have same dim counts, or outer-most dim needs to be 1: {ops[0]} vs {ops[1]}"
        ops1_shape = [1] + ops1_shape

    broadcast = []
    for dim in range(dim_r, dim_r+2):
        if ops[0][dim] != ops[1][dim]:
            if ops[1][dim] == TILE_DIM:
                broadcast.append((1, dim, ops[0][dim]))
                output_shape[dim] = ops[0][dim]
            else:
                assert ops[0][dim] == TILE_DIM, f"Eltwise binary ops must have the same shape in both inputs, or one operand must be 1-tile wide to broadcast: {ops[0]} vs {ops[1]}"
                broadcast.append((0, dim, ops[1][dim]))
                output_shape[dim] = ops[1][dim]

    if output_shape[-2] > tile_height and tile_height != TILE_DIM:
        # Tiny tile
        output_shape[-2] = tile_height
    else:
        output_shape[-2] = align_up(output_shape[-2], tile_height)

    return tuple(output_shape), broadcast


def parallelization(type, attr, op_shape):
    if type in {"binary_vstack", "binary_hstack"}:
        # Unsupported HW op
        return None
    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)


def input_ublock_order(type, attr, num_operands):
    return None


def execution_cycles(type, arch_name, op_model) -> int:
    op_model_desc = op_model_to_desc(type, arch_name, op_model)

    compiler_cache_cycles = get_compiler_cached_cycles(op_model_desc)
    if compiler_cache_cycles is not None:
        return compiler_cache_cycles

    cycle_count = get_op_model_execution_cycles(op_model_desc)
    return cycle_count
