# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentError

import pybuda._C.balancer as balancer
from ..common import to_torch_operands


def eval(type, attr, ops):
    assert len(ops) == 1, "DRAM Queue should have one input"
    t_ops = to_torch_operands(*ops)

    return t_ops[0]

def shape(type, attr, ops, tile_height, tile_width):
    assert len(ops) == 1, "DRAM Queue should have one input"
    return ops[0], []


def parallelization(type, attr, op_shape):
    return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)


def input_ublock_order(type, attr, num_operands):
    return None
