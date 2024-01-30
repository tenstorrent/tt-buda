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
from pybuda.utils import align_up_tile
from pybuda._C.graph import UBlockOrder

from ..common import to_torch_operands, math_fidelity_to_multiplier, data_format_to_int, op_model_to_desc, cast_for_cpu_eval


def eval(type, attr, ops):
    assert len(ops) <= 3, "Depthwise matmul should have two or three inputs"
    assert len(attr) == 1, f"Unexpected number of attrs for depthwise matmul: {len(attr)}"

    t_ops = to_torch_operands(*ops)
    t_ops, original_type = cast_for_cpu_eval(t_ops, type)
    in0 = t_ops[0]
    in1 = t_ops[1]
    bias = t_ops[2] if len(t_ops) == 3 else None

    assert in0.shape[1] == in1.shape[1], "Z dims should match"
    # if in0.shape[1] != in1.shape[1]:
    #     assert in0.shape[1] == 1 or in1.shape[1] == 1, "If z dims don't match, one of them should be 0"
    #     if in0.shape[1] == 1:
    #         in0 = in0.broadcast_to(-1, in1.shape[1], ...)
    #     elif in1.shape[1] == 1:
    #         in1 = in1.broadcast_to(-1, in0.shape[1], ...)

    cnt_kernels = attr[0]
    result = torch.zeros((1, in0.shape[1], in0.shape[2], in1.shape[3]), dtype=torch.float32, requires_grad=False)

    kernel_ratio = in0.shape[3] // in1.shape[2]

    for idx in range(cnt_kernels):
        kernel = in1[:, :, idx * TILE_DIM: (idx + 1) * TILE_DIM, :]
        section_h = idx * kernel_ratio * TILE_DIM
        for idx_ratio in range(kernel_ratio):
            result[..., idx_ratio * TILE_DIM: (idx_ratio + 1) * TILE_DIM] += \
                torch.matmul(in0[..., section_h + idx_ratio * TILE_DIM: section_h + (idx_ratio + 1) * TILE_DIM],
                            kernel[..., idx_ratio * TILE_DIM: (idx_ratio + 1) * TILE_DIM])

    if bias is not None:
        result += bias

    return result.to(original_type)


def shape(type, attr, ops, tile_height, tile_width):
    assert len(ops) in [2, 3], "Depthwise matmul should have two or three inputs"

    output_shape = [ops[0][0], ops[0][1], ops[0][2], ops[1][3]]

    broadcast = []
    if ops[0][1] != ops[1][1]:
        if ops[0][1] == 1:
            broadcast.append((0, 1, ops[1][1]))
            output_shape[1] = ops[1][1]
        elif ops[1][1] == 1:
            broadcast.append((1, 1, ops[0][1]))
            output_shape[1] = ops[0][1]
        else:
            assert False, "If Z dimension is not the same for depthwise, one of operands must have it be 1."

    return tuple(output_shape), broadcast


def parallelization(type, attr, op_shape, fracture_factor):
    return (op_shape.outputs[0].rt, 1)


def input_ublock_order(type, attr, num_operands):
    return [UBlockOrder.C, UBlockOrder.R, UBlockOrder.R]


def execution_cycles(type, arch_name, op_model) -> int:
    op_model_desc = op_model_to_desc(type, arch_name, op_model)
    cycle_count = get_op_model_execution_cycles(op_model_desc)
    return cycle_count
