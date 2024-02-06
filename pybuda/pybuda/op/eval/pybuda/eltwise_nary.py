# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple
from math import gcd
import ast
import os
import torch
import math
import pybuda
from ..common import to_torch_operands
from .transpose import TransposeTM
from .nop import Nop
from .buffer import Buffer
from ..buda.splice import Splice
from pybuda.pybudaglobal import TILE_DIM, align_up_tile, is_tile_dim_aligned
from ..sparse_utils import (
    create_flattened_padding_removal_sparse_picker_matrix,
)
from loguru import logger


def eval(type, attr, ops):

    if type == "conv_sum":

        t_ops = to_torch_operands(*ops)
        
        t_ops = list(t_ops)

        # Extract attributes
        originalY = attr[0]
        originalX = attr[1]
        shifts = attr[2:]

        # Check operands
        for t_op in t_ops:
            assert len(t_op.shape) == 4, f'Tensor must have 4 dimensions, given {len(t_op.shape)}'

        # To pybuda shape
        for i in range(len(t_ops)):
            t_ops[i] = t_ops[i][:, :, :originalY*originalX, :]
            t_ops[i] = t_ops[i].transpose(2, 3)
            t_ops[i] = t_ops[i].reshape(1, t_ops[i].shape[2], originalY, originalX)

        # Shift and Add
        res = 0
        for i in range(len(t_ops)):
            res += torch.nn.functional.pad(t_ops[i], (shifts[2 * i], -shifts[2 * i], shifts[2 * i + 1], -shifts[2 * i + 1]))

        # To buda shape
        res = res.reshape(1, res.shape[1], res.shape[2] * res.shape[3], 1)
        res = res.transpose(1, 3)

        return res

    elif type == "concatenate":
        assert len(attr) == 1, "Concatenate should have 1 attr"
        t_ops = to_torch_operands(*ops)
        return torch.cat(t_ops, dim=attr[0])

    elif type == "where":
        return torch.where(ops[0].type(torch.bool), ops[1], ops[2])

    elif type == "index_copy":
        t_ops = to_torch_operands(*ops)
        out = t_ops[0].index_copy(attr[0], t_ops[1], t_ops[2])
        return out

    elif type == "stack":
        assert len(attr) == 1, "Stack should have 1 attr"
        t_ops = to_torch_operands(*ops)
        return torch.stack(t_ops, dim=attr[0])

    elif type == "interleave":
        assert len(attr) == 2, "Interleave should have 2 attr"
        axis = attr[0]
        stride = attr[1]
        assert axis == -3 and stride == 1

        # Forward impl only works for Z dim interleave with stride 1
        t_ops = to_torch_operands(*ops)
        stacked = torch.stack(t_ops, dim=axis)
        target_shape = list(t_ops[0].shape)
        for op in t_ops[1:]:
            target_shape[axis] += op.shape[axis]

        return torch.reshape(stacked, target_shape)
    assert False, f"Unknown eval: {type}"


# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops) -> Tuple[Tuple, List]:
    def get_eltwise_shape_and_broadcast():
        broadcast = []
        output_dims = max(len(op) for op in ops)
        for index in range(len(ops)):
            ops[index] = list(ops[index])
            if len(ops[index]) < output_dims:
                ops[index] = [1] * (output_dims - len(ops[index])) + ops[index]

        output_shape = [max(dim) for dim in zip(*ops)]
        for op_index in range(len(ops)):
            for dim_index in range(len(ops[op_index])):
                if ops[op_index][dim_index] != output_shape[dim_index]:
                    assert ops[op_index][dim_index] == 1, f"Eltwise nary ops must have same shape or operand must be 1 wide to broadcast: {ops}"
                    broadcast.append((op_index, dim_index - len(output_shape), output_shape[dim_index]))
        
        return tuple(output_shape), broadcast

    if type == "conv_sum":
        shapes = []
        for op in ops:
            assert len(op) <= 4, "Shape of an operand must be smaller than or equal to 4"
            if len(op) < 4:
                op = (4 - len(op)) * (1, ) + op
            if len(shapes) > 0:
                assert shapes[-1] == op, "Shapes of all operands must be the same size"
            shapes.append(op)

        return shapes[0], []

    elif type == "concatenate":
        assert len(attr) == 1, "Concatenate should have 1 attr"
        axis = attr[0]

        output_shape = list(ops[0])
        for op in ops[1:]:
            output_shape[axis] += op[axis]
        
        return output_shape, []

    elif type == "where":
        return get_eltwise_shape_and_broadcast()

    elif type == "index_copy":
        return get_eltwise_shape_and_broadcast()

    elif type == "stack":
        axis = attr[0]

        output_shape = list(ops[0])
        output_shape.insert(axis, len(ops))
        return output_shape, []

    elif type == "interleave":
        assert len(attr) == 2, "Interleave should have 2 attr"
        axis = attr[0]
        stride = attr[1]
        assert axis == -3 and stride == 1

        output_shape = list(ops[0])
        for op in ops[1:]:
            output_shape[axis] += op[axis]
        
        return output_shape, []
    assert False, f"{type} not defined in eltwise_nary"


def lower(type, attr, lc, ops, outputs):

    if type == "conv_sum":
        return lc.op(type, ops, attr, {"a": 0})

    elif type == "concatenate":
        assert len(attr) == 1, "Concatenate should have 1 attr"
        axis = attr[0]
        return lc.op(Splice.create_concatenate(axis, list(op.shape for op in ops)), ops)

    elif type == "index_copy":
        assert len(attr) == 1, "index_copy should have 1 attr"
        dim = attr[0]

        buda_attrs = {
            "axis" : dim,
        }

        return lc.op("index_copy", ops, attr, buda_attrs)

    elif type == "where":
        assert False, "Where is meant to be removed by consteval"

    elif type == "interleave":
        assert len(attr) == 2, "Interleave should have 2 attr"
        axis, stride = attr
        assert axis == -3 and stride == 1
        return lc.op(Splice.create_interleave(axis, stride, list(op.shape for op in ops)), ops)

    assert False, f"{type} not defined in eltwise_nary"


def backward(op_type, attr, ac, operand, inputs, output, grad):
    if op_type == "conv_sum":
        y = attr[0]
        x = attr[1]
        shifts = attr[2:]

        return ac.op("conv_sum", [grad], [y, x, -shifts[operand * 2], -shifts[operand * 2 + 1]])

    elif op_type == "concatenate":
        axis = attr[0]
        dim_offset = grad.shape[axis] 

        index_offset = 0
        for (i, input_) in enumerate(inputs):
            if operand is not i:
                index_offset += input_.shape[axis]
                continue
            return ac.op("select", (grad, ), (axis, index_offset, input_.shape[axis], dim_offset))            
 
    elif op_type == "interleave":
        axis = attr[0]
        stride = attr[1]
        assert axis == -3 and stride == 1

        num_operands = len(inputs)
        result = grad
        if grad.shape[-1] % TILE_DIM != 0:
            result = ac.op("pad_tile", (result, ), (-1, grad.shape[-1]))
        if grad.shape[-2] % TILE_DIM != 0:
            result = ac.op("pad_tile", (result, ), (-2, grad.shape[-2]))
        result = ac.op("hstack", (result, ), (num_operands,))
        if grad.shape[-2] % TILE_DIM != 0:
            result = ac.op("narrow", (result, ), (-2, 0, grad.shape[-2], result.shape[-2]))
        result = ac.op("select", (result, ), (-1, operand*align_up_tile(grad.shape[-1]), align_up_tile(grad.shape[-1]), result.shape[-1]))
        if grad.shape[-1] % TILE_DIM != 0:
            result = ac.op("narrow", (result, ), (-1, 0, grad.shape[-1], result.shape[-1]))
        return result

    assert False, f"{op_type} not defined in eltwise_nary"

def decompose(type, attr, dc, inputs):
    if type == "stack":
        assert len(attr) == 1, "Stack should have 1 attr"
        axis = attr[0]

        new_inputs = []
        for inp in inputs:
            inp_shape = inp.shape.as_list()
            inp_shape.insert(axis, 1)
            new_inp = dc.op("reshape", [inp], (*inp_shape,))
            new_inputs.append(new_inp)

        output = dc.op("concatenate", new_inputs, (axis,))
        dc.fuse(output)
        
    if type == "concatenate":
        if len(inputs) == 1:
            dc.fuse(dc.op(Nop.create(), [inputs[0]]))
    
from math import gcd
from functools import reduce
def find_gcd(list):
    x = reduce(gcd, list)
    return x


def decompose_post_optimize(type, attr, dc, inputs):
    if type == "concatenate":
        axis = attr[0]
        in1 = inputs[0]

        if axis >= 0:
            axis -= len(in1.shape)

        if len(inputs) == 1:
            result = dc.op(Nop.create(), [in1])
            dc.fuse(result)
            return

        # if axis == -1 and len(inputs) == 2 and all([inp.shape.as_list()[axis] < TILE_DIM  for inp in inputs]):
        #     pass

        # maximum number of inputs is 8
        max_inputs = int(os.environ.get("PYBUDA_MAX_CONCAT_INPUTS", "8"))
        if len(inputs) > max_inputs:
            # TODO: use max_num_inputs 8 when tenstorrent/pybuda#316 is resolved
            max_num_inputs = min(6, max_inputs)
            idx = 0
            concats = []
            result = dc.op("concatenate", inputs[idx:idx+max_num_inputs], (axis, ))
            idx += max_num_inputs
            while (idx < len(inputs)):
                result = dc.op("concatenate", [result] + inputs[idx:idx + (max_num_inputs-1)], (axis, ))
                idx += (max_num_inputs-1)
            dc.fuse(result)
            return

        if (
            int(os.environ.get("PYBUDA_CONCAT_SLICE_Y", "0")) != 0
            and axis == -1 and inputs[0].shape[-2] > 10000
            and inputs[0].shape[-2] % TILE_DIM == 0
            and all([len(inp.shape) >= 2 for inp in inputs])
            and all([inp.shape[-2] == in1.shape[-2] for inp in inputs])
        ):
            num_split_r = int(os.environ.get("PYBUDA_CONCAT_SLICE_Y", "0"))
            rows_per_split = inputs[0].shape[-2] // num_split_r

            concats = []
            for i in range(num_split_r):
                split_inputs = []
                for j in range(len(inputs)):
                    split_inputs.append(dc.op("select", [inputs[j]], (-2, rows_per_split*i, rows_per_split, inputs[j].shape[-2])))

                current_concat = dc.op("concatenate", split_inputs, (axis,))
                concats.append(current_concat)

            result = dc.op("concatenate", concats, (-2,))
            dc.fuse(result)
            return

        padding_removal_needed = False

        inputs_unsqueezed = False
        if len(inputs[0].shape) == 1:
            inputs_unsqueezed = True
            for i, inp in enumerate(inputs):
                inputs[i] = dc.op("unsqueeze", [inp], (0, 1))

        if axis > -3:
            for operand in inputs:
                if operand.shape[axis] % TILE_DIM:
                    padding_removal_needed = True

        # Insert slice only when concat on last dim && requires sparse matmul
        insert_slice = (padding_removal_needed and axis == -1) or bool(int(os.environ.get("PYBUDA_INSERT_SLICE_FOR_CONCAT", "0")))
        if insert_slice:
            length_at_dim = [inp.shape[axis] for inp in inputs]
            # large concats on x&y need to be sliced and streamed
            concat_slice_dim = None
            if axis >= -2 and sum(length_at_dim) > 8192:
                concat_slice_dim = "v" if axis == -2 else "h"
                divisor = find_gcd([math.ceil(inp.shape[axis] / TILE_DIM) for inp in inputs])
                slice_size = TILE_DIM * divisor

            non_concat_slice_dim = None
            if axis >= -2 and len(inputs[0].shape) >= 2:
                non_concat_dim = -1 if axis == -2 else -2
                if inputs[0].shape[non_concat_dim] > 8192:
                    non_concat_slice_dim = "v" if non_concat_dim == -2 else "h"

        cols = []
        padded_shape_len = 0
        unpadded_shape_len = 0
        if axis > -3:
            padded_inputs = []
            for operand in inputs:
                unpadded_shape_len += operand.shape[axis]
                cols.extend(torch.arange(0, operand.shape[axis]) + padded_shape_len)
                if operand.shape[axis] % TILE_DIM:
                    padded_inputs.append(dc.op("pad_tile", [operand], (axis, operand.shape[axis])))
                else:
                    padded_inputs.append(operand)
                    
                padded_shape_len += padded_inputs[-1].shape[axis]

                if insert_slice and concat_slice_dim is not None:
                    padded_inputs[-1] = dc.op(Buffer.create(), [padded_inputs[-1], ])
                    assert padded_inputs[-1].shape[axis] % slice_size == 0
                    padded_inputs[-1] = dc.op(concat_slice_dim + "slice", [padded_inputs[-1], ], (padded_inputs[-1].shape[axis] // slice_size , ))
                elif insert_slice and non_concat_slice_dim is not None:
                    slices = inputs[0].shape[non_concat_dim] // TILE_DIM
                    padded_inputs[-1] = dc.op(Buffer.create(), [padded_inputs[-1], ])
                    padded_inputs[-1] = dc.op(non_concat_slice_dim + "slice", [padded_inputs[-1], ], (slices, ))
                

            # we only want to do padding removal once, don't further decompose
            if insert_slice and concat_slice_dim is not None:
                result = dc.op("concatenate", padded_inputs, (-3, ), copy_tms=True, dont_decompose=True)
                assert result.shape[-3] % inputs[0].shape[-3] == 0
                result = dc.op(concat_slice_dim + "stack", [result], (result.shape[-3] // inputs[0].shape[-3], ))
            else:
                result = dc.op("concatenate", padded_inputs, (axis, ), copy_tms=True, dont_decompose =True)

            if padding_removal_needed:
                narrow_unpadded_shape_len = unpadded_shape_len
                unpadded_shape_len = align_up_tile(unpadded_shape_len)
                if axis == -1:
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                cols = torch.tensor(cols)
                rows = torch.arange(len(cols))
                pad_for_factorization = False
                sparse_r_padding = ast.literal_eval(os.environ.get('PYBUDA_PAD_SPARSE_MM', "{}"))
                sparse_r = unpadded_shape_len // 32
                if sparse_r in sparse_r_padding:
                    pad_for_factorization = True
                    splice_unpadded_shape_len = unpadded_shape_len
                    unpadded_shape_len = sparse_r_padding[sparse_r] * 32

                spm = torch.sparse_coo_tensor(
                    [rows.tolist(), cols.tolist()],
                    torch.ones(cols.shape[0]),
                    (unpadded_shape_len, padded_shape_len),
                    dtype=torch.float32,
                )

                # Make W dim 1 for sparse_matmul
                if len(result.shape) > 3 and result.shape[-4] != 1:
                    result = dc.op("reshape", [result], (1, result.shape[-3] * result.shape[-4], result.shape[-2], result.shape[-1]))
                if len(result.shape) >= 3:
                    spm = torch.unsqueeze(spm, 0)
                    spm = torch.stack([spm] * result.shape[-3], -3)
                    if len(result.shape) == 3:
                        spm = torch.squeeze(spm.to_dense(), 0).to_sparse()

                lhs = dc.tensor(spm)
                result = dc.op("sparse_matmul", [lhs, result])
                if pad_for_factorization:
                    result = dc.op("select", [result], (-2, 0, splice_unpadded_shape_len, lhs.shape[-2]))
                    unpadded_shape_len = splice_unpadded_shape_len
                result = dc.op("narrow", [result], (-2, 0, narrow_unpadded_shape_len, unpadded_shape_len))


                if axis == -1:
                    result = dc.op(TransposeTM.create(-2, -1), [result])

            if insert_slice and non_concat_slice_dim is not None:
                result = dc.op(non_concat_slice_dim + "stack", [result], (result.shape[-3] // inputs[0].shape[-3], ))

            if inputs_unsqueezed:
                result = dc.op("squeeze", [result, ], (0,))
            dc.fuse(result)

    elif type == "where":
        
        condition = inputs[0]
        x = inputs[1]
        y = inputs[2]
        one = dc.tensor(torch.ones((1,)))
        not_condition = dc.op("subtract", [one, condition])

        t0 = dc.op("multiply", [condition, x])
        t1 = dc.op("multiply", [not_condition, y])

        add = dc.op("add", [t0, t1])
        dc.fuse(add)
