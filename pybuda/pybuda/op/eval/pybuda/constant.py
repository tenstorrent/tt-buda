# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional

def shape(type, attr, ops):
    assert len(ops) == 0, "constant should not have any operands"
    assert len(attr) == 1, "constant should contain single attr repr the const. val"
    return [1], []


def eval(type, attr, ops):
    assert len(ops) == 0, "constant should not have any operands"
    assert len(attr) == 1, "constant should contain single attr repr the const. val"

    # TODO: add data format
    const_tensor = torch.zeros([1])
    const_tensor[0] = attr[0]

    return const_tensor
