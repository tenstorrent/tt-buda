# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional
from ..interface import PyEltwiseUnaryOp
from loguru import logger
from ..common import to_torch_operands
from ....pybudaglobal import TILE_DIM
from ....tensor import buda_dataformat_to_pytorch_dtype
import numpy as np
from pybuda.op.eval.common import calculate_tile_size
from ..buda.nop import Nop as BudaNop


class Nop(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("nop")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "nop should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = tensors[0]

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])
        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Eltwise unary should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Nop should have one input"
        assert operand == 0, "Invalid operand index"
        return ac.op(Nop.create(), [grad])

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1, "Nop should  have one input"

        if bool(int(os.environ.get("PYBUDA_ENABLE_TINY_TILE", "0"))):
            node_shape = list(tensors[0].shape)
            tile_height = calculate_tile_size(node_shape[-2])
            tile_width = calculate_tile_size(node_shape[-1])
        else:
            tile_height, tile_width = TILE_DIM, TILE_DIM

        lc.op(BudaNop.create(), tensors, tile_height=tile_height, tile_width=tile_width)
