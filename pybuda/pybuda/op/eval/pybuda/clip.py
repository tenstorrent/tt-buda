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


class Clip(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, min=float('-inf'), max=float('inf')):
        self = cls("clip")
        self.min = min
        self.max = max

        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Clip should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.clip(tensors[0], min=self.min, max=self.max)

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Clip should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Clip should have one input"
        assert operand == 0, "Invalid operand index"
        heaviside = ac.op("heaviside", (inputs[0], ac.constant(0.5)))
        subtract = ac.op("subtract", (heaviside, ac.constant(0.5)))
        stretched = ac.op("multiply", (subtract, ac.constant(2.0)))
        return ac.op("multiply", (stretched, grad))

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1, "Clip should  have one input"

        min_value = self.min
        max_value = self.max

        # Inf protection
        if max_value > 65504.0:
            max_value = 65504.0

        if (min_value == 0) and (max_value >= 0):
            res = lc.op(BudaNop.create(relu_en=True, relu_threshold=max_value, relu_mode="max"),(tensors[0],))
            return

        shape = list(tensors[0].shape.as_list())
        # Align up to tile
        shape[-2] = ((shape[-2] - 1) // TILE_DIM + 1) * TILE_DIM
        shape[-1] = ((shape[-1] - 1) // TILE_DIM + 1) * TILE_DIM
        # Align up to 4 dimensions
        if len(shape) > 4:
            raise RuntimeError(
                "Operator clip, operand must have number of dimensions less or equal to 4. "
            )
        if len(shape) < 4:
            shape = (4 - len(shape)) * [1] + shape

        min_value_tensor = lc.tensor(torch.zeros(shape) + min_value)
        max_value_tensor = lc.tensor(torch.zeros(shape) + max_value)
        diff_tensor = lc.tensor(torch.zeros(shape) + max_value - min_value)

        # General Formula/Algorithm
        # y = ReLU(x - min_value) + min_value
        # y = ReLU(0.0 - y + max_value) - max_value
        # y = 0.0 - y

        res = lc.op("subtract", (tensors[0], min_value_tensor))
        # x - min_value
        res = lc.op(BudaNop.create(relu_en=True, relu_threshold=0.0 ,relu_mode="min"),(res,))

        # ReLU(x - min_value)
        res = lc.op("subtract", (diff_tensor, res))
        # diff_value - ReLU(x - min_value), diff = max - min
        res = lc.op(
            BudaNop.create(relu_en=True, relu_threshold=0.0,relu_mode="min"),
            (res,))
        
        # ReLU(diff_value - ReLU(x - min_value))
        lc.op("subtract", (max_value_tensor, res))
        # max_value - ReLU(diff_value - ReLU(x - min_value))

    def backward(self, ac, operand, inputs, output, grad):
        x = inputs[0]
        shape = x.shape.as_list()
        min_value = self.min
        max_value = self.max
        min_value_tensor = ac.tensor(torch.zeros(shape) + min_value)
        max_value_tensor = ac.tensor(torch.zeros(shape) + max_value)

        ge_x = ac.op("greater_equal", (x, min_value_tensor))
        le_x = ac.op("less_equal", (x, max_value_tensor))
        mask = ac.op("multiply", (ge_x, le_x))
        res = ac.op("multiply", (mask, grad))
        return res
