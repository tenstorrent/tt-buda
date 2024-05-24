# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple

import torch
import pybuda
import pybuda._C.balancer as balancer
from pybuda._C.backend_api import get_op_model_execution_cycles
from ....pybudaglobal import TILE_DIM
from ..common import to_torch_operands, op_model_to_desc, get_compiler_cached_cycles
from pybuda.utils import align_up_tile, round_up_div
from .tm import eval as tm_eval


def eval(type, attr, ops):
    if type == "conv_sum":

        t_ops = to_torch_operands(*ops)
        
        t_ops = list(t_ops)

        # Extract attributes
        originalY = attr[0]
        originalX = attr[1]
        shifts = attr[2:]

        # Check operands
        for t_op in t_ops:
            assert len(t_op.shape) == 4, f'Tensor must have 4 dimensions, given {len(t_op.shape)}'

        # To pybuda shape
        for i in range(len(t_ops)):
            t_ops[i] = t_ops[i][:, :, :originalY*originalX, :]
            t_ops[i] = t_ops[i].transpose(2, 3)
            t_ops[i] = t_ops[i].reshape(1, t_ops[i].shape[2], originalY, originalX)

        # Shift and Add
        res = 0
        for i in range(len(t_ops)):
            res += torch.nn.functional.pad(t_ops[i], (shifts[2 * i], -shifts[2 * i], shifts[2 * i + 1], -shifts[2 * i + 1]))

        # To buda shape
        res = res.reshape(1, res.shape[1], res.shape[2] * res.shape[3], 1)
        res = res.transpose(1, 3)

        pad = align_up_tile(res.shape[2]) - res.shape[2]
        if pad > 0:
            res = torch.nn.functional.pad(res, (0, 0, 0, pad))

        return res
   

    if type == "concatenate" or type == "hconcat" or type == "vconcat":
        assert len(ops) + 1 == len(attr), f"{len(ops)} {len(attr)}"
        axis = attr[0]
        unpaded_ops = []
        for op, size in zip(ops, attr[1:]):
            indices = torch.arange(size)
            unpaded = torch.index_select(op, axis, indices)
            t_op = to_torch_operands(unpaded)
            unpaded_ops.append(t_op[0])

        unpaded_result = torch.cat(unpaded_ops, axis=axis)
        r = unpaded_result.shape[-2]
        c = unpaded_result.shape[-1]
        pad_r = align_up_tile(r)
        pad_c = align_up_tile(c)
        return torch.nn.functional.pad(unpaded_result, (0, pad_c - c, 0, pad_r - r))

    if type == "index_copy":

        t_ops = to_torch_operands(*ops)
        out = t_ops[0].index_copy(attr[0], t_ops[1][0, 0, 0, :], t_ops[2])
        return out

    assert False, f"Unknown eval: {type}"


# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops, tile_height, tile_width) -> Tuple[Tuple, List]:
    if type == "conv_sum":
        shapes = []
        for op in ops:
            assert len(op) <= 4, "Shape of an operand must be smaller than or equal to 4"
            if len(op) < 4:
                op = (4 - len(op)) * (1, ) + op
            if len(shapes) > 0:
                assert shapes[-1] == op, "Shapes of all operands must be the same size"
            shapes.append(op)

        return shapes[0], []

    if type == "concatenate" or type == "hconcat" or type == "vconcat":
        axis = attr[0]

        output_shape = list(ops[0])
        output_shape[axis] = 0
        for size in attr[1:]:
            output_shape[axis] += size

        if axis > 1 or (axis < 0 and axis + len(output_shape) > 1):
            output_shape[axis] = align_up_tile(output_shape[axis])
        return output_shape, []

    if type == "index_copy":
        return ops[0], []
        
    assert False, f"{type} not defined in eltwise_nary"


def parallelization(type, attr, op_shape):
    return None


def input_ublock_order(type, attr, num_operands):
    return None


def execution_cycles(type, arch_name, op_model) -> int:
    op_model_desc = op_model_to_desc(type, arch_name, op_model)

    compiler_cache_cycles = get_compiler_cached_cycles(op_model_desc)
    if compiler_cache_cycles is not None:
        return compiler_cache_cycles

    if type in ["conv_sum", "concatenate", "hconcat", "vconcat", "index_copy"]:
        # TODO not yet implemented
        cycle_count = 0
    else:
        assert False, f"Unknown execution_cycles: {type}"

    return cycle_count
