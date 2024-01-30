# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch

from pybuda.pybudaglobal import TILE_DIM
from ..common import to_torch_operands, cast_for_cpu_eval


def eval(type, attr, ops):
    assert type == "depthwise", f"Unknown type {type} in depthwise matmul"
    assert len(ops) in [2, 3], "Depthwise matmul should have two or three inputs"
    assert len(attr) == 1, "Depthwise matmul should have one attribute"

    t_ops = to_torch_operands(*ops)
    t_ops, original_type = cast_for_cpu_eval(t_ops, type)
    in0 = t_ops[0]
    in1 = t_ops[1]
    bias = t_ops[2] if len(t_ops) == 3 else None

    assert in0.shape[1] == in1.shape[1] == 1, "Z dim should be 1"  # todo: allow this

    cnt_kernels = attr[0]
    result = torch.zeros((1, 1, in0.shape[2], in1.shape[3]), dtype=torch.float32, requires_grad=False)

    kernel_ratio = in0.shape[3] // in1.shape[2]

    for idx in range(cnt_kernels):
        kernel = in1[:, :, idx * TILE_DIM: (idx + 1) * TILE_DIM, :]
        section_h = idx * kernel_ratio * TILE_DIM
        for idx_ratio in range(kernel_ratio):
            result[..., idx_ratio * TILE_DIM: (idx_ratio + 1) * TILE_DIM] += \
                torch.matmul(in0[..., section_h + idx_ratio * TILE_DIM: section_h + (idx_ratio + 1) * TILE_DIM],
                             kernel[..., idx_ratio * TILE_DIM: (idx_ratio + 1) * TILE_DIM])

    assert bias is None, "Unexpected fused bias in depthwise, can be added..."

    return result.to(original_type)


def shape(type, attr, ops):
    assert len(ops) in [2, 3], "Depthwise matmul should have two or three inputs"
    assert len(attr) == 1, "Depthwise matmul should have one attribute"

    ops[0] = list(ops[0])
    ops[1] = list(ops[1])

    assert len(ops[0]) == 4 and len(ops[0]) == 4
    assert ops[0][3] % ops[1][2] == 0, "Number of kernel points doesn't divide in0's inner dim"
    assert ops[0][1] == ops[0][1] == 1, "Unexpected z dim > 1"  # todo: allow this

    output_dim = [1, 1, ops[0][2], ops[1][3]]

    return output_dim, []


def lower(type, attr, lc, ops, outputs):
    assert len(ops) in [2, 3], "Depthwise matmul should have two or three inputs"
    # assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    buda_attrs = {}

    if len(ops) == 3:
        buda_attrs["bias"] = True

    lc.op(type, ops, attr, buda_attrs)

def decompose(type, attr, dc, inputs):
    pass


def backward(type, attr, ac, operand, inputs, output, grad):
    assert False, "not yet implemented"
