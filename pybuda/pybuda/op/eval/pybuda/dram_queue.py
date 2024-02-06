# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional
from .buffer import Buffer

from ..common import to_torch_operands

def eval(type, attr, ops):
    assert len(ops) == 1, "DRAM Queue should have one input"
    t_ops = to_torch_operands(*ops)

    return t_ops[0]


def shape(type, attr, ops):
    assert len(ops) == 1, "DRAM Queue should have one input"
    return ops[0], []


def lower(type, attr, lc, ops, outputs):
    lc.op(type, ops, attr)


def backward(type, attr, ac, operand, inputs, output, grad):
    return ac.op(Buffer.create(), (grad, ))
