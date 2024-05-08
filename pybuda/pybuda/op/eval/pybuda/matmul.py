# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import typing
from math import sqrt
import os

import pybuda._C.balancer as balancer
from pybuda._C import DataFormat
import torch

from pybuda.pybudaglobal import TILE_DIM
from ..common import to_torch_operands, cast_for_cpu_eval
from ..sparse_utils import transpose_sparse_picker_matrix, create_sparse_buda, shapeify_sparse_tiles_and_encodings, is_kernel_fracturing_candidate
from pybuda.utils import round_up_div
from pybuda.op.eval.common import calculate_tile_size
from .transpose import TransposeTM

def eval(type, attr, ops):
    assert len(ops) in [2, 3], "Matrix multiply should have two or three inputs"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    accumulate = (len(attr) >= 1) and bool(attr[0])
    t_ops = to_torch_operands(*ops)
    t_ops, original_type = cast_for_cpu_eval(t_ops, type)

    if type == "matmul":
        result = torch.matmul(t_ops[0], t_ops[1])
        result = result.to(original_type)
        if len(t_ops) > 2:
            result += t_ops[2] # bias
    elif type == "sparse_matmul":
        a = t_ops[0]
        b = t_ops[1]

        assert len(t_ops) == 2, "Sparse matmul can't have a fused bias"
        assert a.is_sparse
        if len(a.shape) == 2:
            if len(b.shape) == 2:
                return torch.sparse.mm(a, b)
            else:
                has_w = len(b.shape) == 4
                if has_w:
                    b = b.squeeze(0)

                if b.shape[-3] != 1:
                    bcast_amount = b.shape[-3]
                    a = torch.stack([a] * bcast_amount)
                else:
                    a = a.unsqueeze(0)

                result = torch.bmm(a, b)

                if has_w:
                    result = result.unsqueeze(0)
        else:
            assert len(a.shape) >= 3
            assert a.shape[-3] == 1 or b.shape[-3] == 1 or b.shape[-3] == a.shape[-3]
            has_w = len(a.shape) == 4
            while len(a.shape) < 4:
                a = a.unsqueeze(0)

            while len(b.shape) < 4:
                b = b.unsqueeze(0)
            
            if a.shape[-3] == 1:
                bcast_amount = b.shape[-3]
                a = torch.cat([a] * bcast_amount, dim=-3)
            elif b.shape[-3] == 1:
                broadcast_shape = list(b.shape)
                broadcast_shape[-3] = a.shape[-3]
                b = torch.broadcast_to(b, broadcast_shape)
            else:
                assert b.shape[-3] == a.shape[-3]

            if has_w:
                w = a.shape[-4]
                results = []
                for i in range(w):
                    results.append(torch.bmm(a[i], b[i]))
                result = torch.stack(results)
            else:
                result = torch.bmm(a[0], b[0])

        result = result.to(original_type)

    if accumulate and len(result.shape) >= 3:
        result = torch.sum(result, dim=-3, keepdim=True)

    return result


def shape(type, attr, ops):
    assert len(ops) in [2, 3, 4], "Matrix multiply should have two or three inputs"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    accumulate = (len(attr) >= 1) and bool(attr[0])

    ops0_padding = 0
    ops[0] = list(ops[0])
    while len(ops[0]) < len(ops[1]):
        ops[0] = [1] + ops[0]
        ops0_padding += 1

    ops[1] = list(ops[1])
    ops1_padding = 0
    while len(ops[1]) < len(ops[0]):
        ops[1] = [1] + ops[1]
        ops1_padding += 1

    broadcast = []
    output_dim = []
    for dim in range(4, len(ops[0]) + 1):
        assert ops[0][-dim] == ops[1][-dim], f"Broadcast on dimensions beyond 3rd is not supported {ops} {dim}"
        output_dim.append(ops[0][-dim])

    # Z broadcast
    if len(ops[0]) >= 3:
        if (ops[0][-3] != ops[1][-3]):
            if ops[0][-3] == 1:
                broadcast.append((0, len(ops[0]) - 3, ops[1][-3]))
                output_dim.append(ops[1][-3])
            elif ops[1][-3] == 1:
                if type != "sparse_matmul":
                    # Sparse matmul can automatically handle broadcast in this case
                    broadcast.append((1, len(ops[0]) - 3, ops[0][-3]))
                output_dim.append(ops[0][-3])
            else:
                assert False, "If Z dimension is not the same for matmul, one of operands must have it be 1."
        else:
            output_dim.append(ops[0][-3])

    # Inner dim broadcast
    if ops[0][-1] != ops[1][-2]:
        if ops[0][-1] == 1:
            broadcast.append((0, len(ops[0]) - 1 - ops0_padding, ops[1][-2]))
        elif ops[1][-2] == 1:
            broadcast.append((1, len(ops[0]) - 2 - ops1_padding, ops[0][-1]))
        else:
            if type == "sparse_matmul":
                assert ops[0][-1] == ops[1][-2] * ops[1][-3], "Inner dimensions don't match for sparse matmul."
            else:
                assert False, f"If inner dimension is not the same for matmul, one of operands must have it be 1, shapes are: {ops}"

    output_dim.extend([ops[0][-2], ops[1][-1]])
    if accumulate:
        assert len(output_dim) >= 3
        output_dim[-3] = 1

    return output_dim, broadcast

def lower(type, attr, buda_attr, lc, ops, outputs):
    assert len(ops) in [2, 3, 4], "Matrix multiply should have two or three inputs"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes (accumulate, z_bcast_factor, zero_point)"
    has_requant = "requant" in buda_attr and buda_attr['requant']

    accumulate = (len(attr) >= 2) and bool(attr[0]) if has_requant else (len(attr) >= 1) and bool(attr[0])

    buda_attrs = {}
    if 'sfpu_op' in buda_attr and os.environ.get("PYBUDA_FUSE_MATMUL_GELU", "0") != "0":
        buda_attrs["sfpu_op"] = "gelu"
    if accumulate:
        buda_attrs["accumulate"] = True

    if has_requant:
        buda_attrs["requant"] = True
        buda_attrs["zero_point"] = attr[-1]
        if len(ops) == 4:
            buda_attrs["bias"] = True
    else:
        if len(ops) == 3:
            buda_attrs["bias"] = True

    if type == "sparse_matmul":
        in0 = ops[0]
        in1 = ops[1]

        picker = lc.get_pytorch_tensor(in0)
        zdim = 1 if len(picker.shape) < 3 else picker.shape[-3]

        z_bcast_factor = 1 if len(attr) < 2 else attr[1]  # set in sparse matmul's decompose

        # We can fully fracture kH * kW
        max_fracture_factor = z_bcast_factor if is_kernel_fracturing_candidate(ops, z_bcast_factor) else 1

        # # TODO: this shouldn't be a sqrt but kW, though we don't have that info here currently
        # fracture_factor = int(sqrt(z_bcast_factor)) if is_kernel_fracturing_candidate(ops, z_bcast_factor) else 1

        sparse_buda = create_sparse_buda(picker, z_bcast_factor, max_fracture_factor)

        # Set grid_r to smallest valid solution (MaxUblocksR)
        # Hardcode most of the values to 1, potentially add some solvers to choose valid combos if some limitations hit
        u_rt = 1
        u_kt = 1
        u_ct = 1
        t_factor_r = 1
        t_factor_c = 1
        fracture_factor = 1
        grid_r = round_up_div(picker.shape[-2], TILE_DIM)
        grid_c = 1  # this is always 1 by default, before balancing, needed for buda eval

        sparse_tile_ptr_bits = sparse_buda.get_sparse_tile_ptr_bits(grid_r, t_factor_r, u_rt)
        sparse_ublock_idx_bits = sparse_buda.get_sparse_ublock_idx_bits(grid_r, t_factor_r, u_rt)
        sparse, encodings, _s_shape, _e_shape, _num_strips = sparse_buda.get_sparse_tiles_and_encodings(grid_r)
        sparse, encodings = shapeify_sparse_tiles_and_encodings(
            sparse=sparse,
            encodings=encodings,
            grid_r=grid_r,
            fracture_factor=fracture_factor
        )

        sparse_is_binary = (sparse.numel() == (torch.sum(sparse == 1) + torch.sum(sparse == 0))).item()
        sparse_is_int = (sparse.numel() == (torch.sum(sparse == 1) + torch.sum(sparse == 0))).item() and (ops[0].output_df == DataFormat.Int8 or ops[0].output_df == DataFormat.Int32)

        if sparse_is_int:
            target_df = DataFormat.Int8
        elif sparse_is_binary:
            target_df = DataFormat.Bfp2_b
        else:
            target_df = DataFormat.Float16_b

        in0 = lc.tensor_with_sparse_buda(sparse, sparse_buda, target_df)
        in2 = lc.tensor(encodings, DataFormat.RawUInt32)

        is_sparse = True
        buda_attrs["identity"] = True
        buda_attrs["num_sparse_tiles"] = sparse.shape[-1] // TILE_DIM
        buda_attrs["num_index_tiles"] = encodings.shape[-1] // TILE_DIM
        buda_attrs["sparse_tile_ptr_bits"] = sparse_tile_ptr_bits
        buda_attrs["sparse_ublock_idx_bits"] = sparse_ublock_idx_bits
        buda_attrs["fracture_factor"] = fracture_factor
        # We need fracture_factor in attributes as well, since shape() function doesn't get buda attrs
        lc.op("matmul", [in0, in1, in2], (accumulate, is_sparse, sparse_tile_ptr_bits, 1, zdim, picker.shape[-2], in1.shape[-1], fracture_factor, u_rt, u_kt, u_ct, grid_c, t_factor_r, t_factor_c, sparse_ublock_idx_bits), buda_attrs)
    else:
        # Find proper tile sizes
        if bool(int(os.environ.get("PYBUDA_ENABLE_TINY_TILE", "0"))):
            node_shape = lc.pybuda_shape()
            tile_height = calculate_tile_size(node_shape[-2])
            tile_width = TILE_DIM
        else:
            tile_height, tile_width = TILE_DIM, TILE_DIM
        lc.op(type, ops, attr, buda_attrs, "", tile_height, tile_width) # straight 1-1 for matmul


def decompose(type, attr, dc, inputs):
    if type == "sparse_matmul":
        # Special case decomp where RHS bcast over LHS Z dim i.e. in0.z > 1 and in1.z == 1
        # Sparse matmul can handle this case natively and this path enables better streaming
        in0 = inputs[0]
        in1 = inputs[1]
        picker = dc.get_pytorch_tensor(in0)
        zdim = 1 if len(picker.shape) < 3 else picker.shape[-3]

        accumulate = (len(attr) >= 1) and bool(attr[0])
        z_bcast_factor = zdim if (zdim > 1 and in1.shape[-3] == 1) else 1

        # In case of convolutions, z_bcast_factor is the volume of the conv's kernel (kernel_height * kernel_width)

        if z_bcast_factor > 1:
            picker = torch.cat([picker[0][z] for z in range(z_bcast_factor)])
            sparse = dc.tensor(picker)
            result = dc.op("sparse_matmul", [sparse, in1], (accumulate, z_bcast_factor))
            result = dc.op("vslice", [result], (z_bcast_factor,))
            dc.fuse(result)


def backward(type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 2, "Matrix multiply should have two inputs"
    assert operand < 2, "Invalid operand index"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    in0 = inputs[0]
    in1 = inputs[1]

    if type == "sparse_matmul":
        assert operand == 1, "Only support gradients through operand 1"
        in0_value = ac.get_pytorch_tensor(in0)
        assert in0_value.is_sparse
        in0t_value = transpose_sparse_picker_matrix(in0_value)
        in0t = ac.tensor(in0t_value)
        return ac.op("sparse_matmul", (in0t, grad))

    if operand == 0:
        shape_len = len(ac.get_shape(in1))
        in1t = ac.op(TransposeTM.create(-2, -1), [in1])
        return ac.op("matmul", (grad, in1t))

    if operand == 1:
        shape_len = len(ac.get_shape(in0))
        in0t = ac.op(TransposeTM.create(-2, -1), [in0])
        return ac.op("matmul", (in0t, grad))

def initial_flops_estimate(type, attr, ops):
    macc = 0
    if type == "matmul":
        output_shape = shape(type, attr, ops)[0]
        macc = output_shape[-1] * output_shape[-2]
        if len(output_shape) > 2:
            macc *= output_shape[-3]
        if len(output_shape) > 3:
            macc *= output_shape[-4]
        macc *= ops[0][-1]
        
    flops = macc * 2
    return flops
        
