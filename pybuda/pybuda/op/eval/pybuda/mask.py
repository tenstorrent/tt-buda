# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile, round_up_div, clamp
from pybuda import Tensor


def eval(type, attr, ops):

    if type == "mask":
        assert len(attr) == 1

        dim = attr[0]
        inp = ops[0]
        indices = ops[1].long()
        while len(indices.shape) < len(inp.shape):
            indices = indices.unsqueeze(0)
            
        ones = torch.ones(indices.shape, dtype=inp.dtype)
        tensor = torch.zeros(inp.shape, dtype=inp.dtype).scatter_(dim, indices, ones)
        return tensor

def shape(type, attr, ops):
    if type == "mask":
        return ops[0], []


def lower(type, attr, lc, ops, outputs):
    raise RuntimeError("This should never be called.")


def backward(type, attr, ac, operand, inputs, output, grad):
    raise RuntimeError("This should never be called.")

