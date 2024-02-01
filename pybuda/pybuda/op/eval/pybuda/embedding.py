# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..common import to_torch_operands
from pybuda._C import DataFormat
from pybuda._C.graph import RuntimeTensorTransform, RuntimeTensorTransformType
from ....pybudaglobal import TILE_DIM


def eval(type, attr, ops):
    assert type == "embedding"
    assert len(ops) == 2
    t_ops = to_torch_operands(*ops)
    return torch.embedding(t_ops[0], t_ops[1].to(torch.int32))


def shape(type, attr, ops):
    assert type == "embedding"
    assert len(ops) == 2
    shape = list(ops[1])
    shape.append(ops[0][-1])
    return shape, []


def lower(type, attr, lc, ops, outputs):
    assert type == "embedding"
    assert len(ops) == 2

    lc.set_output_df(ops[1], DataFormat.RawUInt32)
    lc.set_runtime_tensor_transform(ops[1], RuntimeTensorTransform.EmbeddingIndex(ops[1].shape))

    embedding_dim = ops[0].shape.as_list()
    while len(embedding_dim) < 4:
        embedding_dim = [1] + embedding_dim

    buda_attrs = {
        "num_indices": ops[1].shape[-1],
    }

    lc.op(type, ops, (ops[1].shape[-1],), buda_attrs, "", TILE_DIM, TILE_DIM)


def decompose(type, attr, dc, inputs):
    pass


def backward(type, attr, ac, operand, inputs, output, grad):
    assert type == "embedding"
    assert len(ops) == 2
    raise NotImplementedError("embedding backwards not implemented")
