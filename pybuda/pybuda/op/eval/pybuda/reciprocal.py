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
from ..buda.reciprocal import Reciprocal as BudaReciprocal


class Reciprocal(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("reciprocal")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Reciprocal should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]

        ret = torch.reciprocal(tensors[0] + 1e-10)  # add epsilon to avoid infinity

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Reciprocal should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1, "Reciprocal should  have one input"
        approximate_mode = True if "PYBUDA_EXP_APPROX" in os.environ else False
        lc.op(BudaReciprocal.create(), tensors)

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Reciprocal should have one input"
        assert operand == 0, "Invalid operand index"
        sq = ac.op("multiply", (output, output))
        neg = ac.op("multiply", (sq, ac.constant(-1)))
        return ac.op("multiply", (neg, grad))
