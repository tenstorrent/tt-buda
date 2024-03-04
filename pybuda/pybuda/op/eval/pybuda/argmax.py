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


class Argmax(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, dim=None):
        self = cls("argmax")
        self.dim = dim

        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Argmax should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors] 

        if hasattr(self, 'dim'):
            dim=self.dim
        else:
            dim=None

        ret = torch.argmax(tensors[0], dim, keepdims=True)

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Argmax should have one input"

        if hasattr(self, 'dim'):
            dim = self.dim
        else:
            dim = None

        if dim is not None:
            shape = list(tensor_shapes[0])
            shape[dim] = 1
        else:
            shape = [1] * len(tensor_shapes[0])
        return tuple(shape), []

    def backward(self, ac, operand, inputs, output, grad):
        raise RuntimeError(
            "Argmax does not require grad and does not have a backwards function"
        )

    def decompose(self, dc, inputs):
        inp_node = inputs[0]

        if hasattr(self, 'dim'):
            axis = self.dim
        else:
            axis=None

        if axis is None:
            import math

            inp_node = dc.op(
                "reshape", [inp_node], (1, math.prod(inp_node.shape.as_list()))
            )
            axis = -1

        input_shape = inp_node.shape.as_list()
        if axis >= 0:
            axis -= len(input_shape)

        data_type = buda_dataformat_to_pytorch_dtype(inp_node.output_df)
        range_shape = [
            dim if i == axis + len(input_shape) else 1
            for i, dim in enumerate(input_shape)
        ]

        range = torch.arange(input_shape[axis], dtype=data_type).reshape(range_shape)
        range_tensor = dc.tensor(range)

        factor = torch.ones((input_shape), dtype=data_type) * 1e10
        factor_tensor = dc.tensor(factor)

        mult_1 = dc.op(
            "multiply",
            [inp_node, factor_tensor],
        )
        softmax = dc.op("softmax", [mult_1], (axis, 1))
        mult_2 = dc.op("multiply", [softmax, range_tensor])
        reduce_sum = dc.op("reduce_sum", [mult_2], (axis,))
        dc.fuse(reduce_sum)

    def lower(self, lc, tensors, outputs):
        return None

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
