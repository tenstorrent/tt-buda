# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional

SINGLE_TILE_SHAPE = (1, 1, 32, 32)

def shape(type, attr, ops, tile_height, tile_width):
    assert len(ops) == 0, "constant should not have any operands"
    assert len(attr) == 1, "constant should contain single attr repr the const. val"
    return SINGLE_TILE_SHAPE, []


def eval(type, attr, ops):
    assert len(ops) == 0, "constant should not have any operands"
    assert len(attr) == 1, "constant should contain single attr repr the const. val"

    # TODO: add data format
    const_tensor = torch.zeros(SINGLE_TILE_SHAPE)
    const_tensor[0, 0, 0, 0] = attr[0]

    return const_tensor
