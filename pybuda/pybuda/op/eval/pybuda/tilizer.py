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
from .nop import Nop

class Tilizer(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("tilizer")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Tilizer should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.tensors[0]

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret
    
    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Tilizer should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Tilizer should have one input"
        assert operand == 0, "Invalid operand index"

        return ac.op(Nop.create(), (grad, ))
    