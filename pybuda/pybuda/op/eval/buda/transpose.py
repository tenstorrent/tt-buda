# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..interface import BudaTM


class TransposeTM(BudaTM):
    @classmethod
    def create(cls, dim0, dim1, z_dim_slice=-1):
        self = cls("transpose")
        self.dim0 = dim0
        self.dim1 = dim1
        self.z_dim_slice = z_dim_slice
        return self

    def eval(self, tensors):
        shape = tensors[0].shape
        assert len(tensors) == 1, "Tensor manipulation ops should have one input"
        assert self.dim0 == -2 or self.dim0 == len(shape) - 2, f"Buda TM transpose can only transpose R/C dims, got dim0: {self.dim0}"
        assert self.dim1 == -1 or self.dim1 == len(shape) - 1, f"Buda TM transpose can only transpose R/C dims, got dim1: {self.dim1}"

        return torch.transpose(tensors[0], self.dim0, self.dim1)

    def shape(self, tensor_shapes, tile_height, tile_width):
        shape = tensor_shapes[0]
        assert len(tensor_shapes) == 1, "Tensor manipulation ops should have one input"
        assert self.dim0 == -2 or self.dim0 == len(shape) - 2, f"Buda TM transpose can only transpose R/C dims, got dim0: {self.dim0}"
        assert self.dim1 == -1 or self.dim1 == len(shape) - 1, f"Buda TM transpose can only transpose R/C dims, got dim1: {self.dim1}"

        (shape[self.dim0], shape[self.dim1]) = (shape[self.dim1], shape[self.dim0])
        return tuple(shape), []
