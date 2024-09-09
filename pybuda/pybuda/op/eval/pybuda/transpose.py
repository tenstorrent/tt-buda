# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..interface import PyTM
from ..buda.transpose import TransposeTM as BudaTransposeTM
from .. import sparse_utils
from pybuda._C import UnsupportedHWOpsError


class TransposeTM(PyTM):
    @classmethod
    def create(cls, dim0, dim1, z_dim_slice=-1):
        self = cls("transpose")
        self.dim0 = dim0
        self.dim1 = dim1
        self.z_dim_slice = z_dim_slice
        return self

    def eval(self, tensors):
        return torch.transpose(tensors[0], self.dim0, self.dim1)

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1
        shape = list(tensor_shapes[0])
        (shape[self.dim0], shape[self.dim1]) = shape[self.dim1], shape[self.dim0]
        return tuple(shape), []

    def backward(self, ac, operand, inputs, output, grad):
        assert operand == 0, "Invalid operand index"
        z_dim_slice = self.z_dim_slice
        if (self.dim0 == -3 and self.dim1 == -4) or (
            self.dim0 == -4 and self.dim1 == -3
        ):
            z_dim_slice = -1
        elif self.dim0 == -3 or self.dim0 == -4:
            z_dim_slice = grad.shape[self.dim1]
        elif self.dim1 == -3 or self.dim1 == -4:
            z_dim_slice = grad.shape[self.dim0]
        else:
            z_dim_slice = -1
        return ac.op(
            TransposeTM.create(self.dim0, self.dim1, z_dim_slice=z_dim_slice),
            [grad],
        )

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1

        if self.dim0 >= 0:
            self.dim0 -= tensors[0].shape.len()
        if self.dim1 >= 0:
            self.dim1 -= tensors[0].shape.len()

        # Buda requires least dim as dim0?
        if self.dim0 > self.dim1:
            tmp = self.dim0
            self.dim0 = self.dim1
            self.dim1 = tmp

        if self.dim0 == -2 and self.dim1 == -1:
            lc.tm(
                BudaTransposeTM.create(self.dim0, self.dim1, z_dim_slice=self.z_dim_slice),
                tensors[0],
            )
        else:
            raise UnsupportedHWOpsError(self)

    def decompose(self, dc, inputs):
        act = inputs[0]
        # canonicalize dims to use negative indexing
        if self.dim0 >= 0 or self.dim1 >= 0:
            if self.dim0 >= 0:
                self.dim0 -= inputs[0].shape.len()
            if self.dim1 >= 0:
                self.dim1 -= inputs[0].shape.len()
            dc.fuse(
                dc.op(
                    TransposeTM.create(
                        self.dim0, self.dim1, z_dim_slice=self.z_dim_slice
                    ),
                    inputs,
                )
            )

    def decompose_post_optimize(self, dc, inputs):
        orig_shape = inputs[0].shape
        if (
            len(orig_shape) > 2
            and (self.dim0 == -3
            and self.dim1 == -1 or self.dim0 == -1
            and self.dim1 == -3)
            and ((len(orig_shape) == 4 and orig_shape[-4] == 1) or len(orig_shape) < 4)
        ):
            # XZ transpose
            result = inputs[0]
            use_sparse_mm = True

            result = inputs[0]
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            result = dc.op(TransposeTM.create(-2, -1), [result])

            if result.shape[-3] > 1:
                result = dc.op("vstack", [result], (orig_shape[-3],))
            i_spm = sparse_utils.create_sparse_interleave_picker_matrix(
                result.shape[-2], orig_shape[-1], orig_shape[-3]
            )
            result = picker_matmul(use_sparse_mm, dc, i_spm, result)

            if orig_shape[-1] > 1:
                result = dc.op("vslice", [result], (orig_shape[-1],))
            result = dc.op(TransposeTM.create(-2, -1), [result])

            result = dc.op(
                "narrow", [result], (-2, 0, orig_shape[-2], result.shape[-2])
            )
            result = dc.op(
                "narrow", [result], (-1, 0, orig_shape[-3], result.shape[-1])
            )

            dc.fuse(result)

        elif (
            self.dim0 == -3
            and self.dim1 == -2
            and ((len(orig_shape) == 4 and orig_shape[0] == 1) or len(orig_shape) == 3)
        ):
            # YZ transpose
            result = inputs[0]
            use_sparse_mm = True

            result = inputs[0]
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))

            if result.shape[-3] > 1:
                result = dc.op("vstack", [result], (orig_shape[-3],))
            i_spm = sparse_utils.create_sparse_interleave_picker_matrix(
                result.shape[-2], orig_shape[-2], orig_shape[-3]
            )
            result = picker_matmul(use_sparse_mm, dc, i_spm, result)

            if orig_shape[-2] > 1:
                result = dc.op("vslice", [result], (orig_shape[-2],))

            result = dc.op(
                "narrow", [result], (-2, 0, orig_shape[-3], result.shape[-2])
            )
            result = dc.op(
                "narrow", [result], (-1, 0, orig_shape[-1], result.shape[-1])
            )

            dc.fuse(result)


def picker_matmul(use_sparse_mm, dc, s, result):
    if use_sparse_mm:
        lhs = dc.tensor(s)
        result = dc.op("sparse_matmul", [lhs, result])
    else:
        lhs = dc.tensor(s.to_dense())
        result = dc.op("matmul", [lhs, result])

    return result
