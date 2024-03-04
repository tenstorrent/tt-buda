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
from ..buda.abs import Abs as BudaAbs
from .nop import Nop


class CumulativeSum(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, axis, exclusive=False):
        self = cls("cumsum")
        self.axis = axis
        self.exclusive = exclusive
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Cumulative Sum should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.cumsum(tensors[0], dim=self.axis)

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Cumulative Sum should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Cumulative Sum should have one input"
        assert operand == 0, "Invalid operand index"
        dim = self.axis
        assert (
            dim == 0
        ), "Unsupported dim different then 0 for cumulative sum backward pass"
        if dim == 0:
            return ac.op(Nop.create(), (grad,))

    def lower(self, lc, tensors, outputs):
        return None

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
