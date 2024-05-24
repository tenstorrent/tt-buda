# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from pybuda._C.graph import UBlockOrder
from ..common import to_torch_operands
from pybuda.tensor import pad_pytorch_tensor_to_buda, align_up_tile


def eval(type, attr, ops):
    assert type == "embedding"
    assert len(ops) == 2
    t_ops = to_torch_operands(*ops)
    num_indices = attr[0]
    indices = t_ops[1].reshape(-1).narrow(0, 0, num_indices)
    table = t_ops[0].squeeze(0).squeeze(0)
    r = torch.embedding(table, indices)
    return pad_pytorch_tensor_to_buda(r, [])


def shape(type, attr, ops, tile_height, tile_width):
    assert type == "embedding"
    assert len(ops) == 2
    num_indices = align_up_tile(attr[0])
    embedding_dim = ops[0][-1]
    shape = [1, 1, num_indices, embedding_dim]
    return shape, []


def parallelization(type, attr, op_shape):
    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)


def input_ublock_order(type, attr, num_operands):
    return [UBlockOrder.R, UBlockOrder.R]


def execution_cycles(type, arch_name, op_model) -> int:
    return 10000
