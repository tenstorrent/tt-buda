# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from ..common import to_torch_operands
import torch
import pybuda
import pybuda._C.balancer as balancer
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile, round_up_div, align_up
from ..sparse_utils import bcast_sparse_picker_matrix


def eval(type, attr, ops):
    assert len(ops) == 1, "Tensor manipulation ops should have one input"
    t_ops = to_torch_operands(*ops)

    dtype = ops[0].dtype
    if type == "transpose":
        assert len(attr) == 3
        dim0, dim1, orig_size = attr
        ret = torch.transpose(t_ops[0], dim0, dim1)
        if orig_size > 0:
            # remove padding when transpose to Z/W dim
            if dim0 == 0:
                ret = ret[:orig_size, :, :, :]
            elif dim0 == 1:
                ret = ret[:, :orig_size, :, :]
            else:
                assert False, "Shouldnt get to this point"
            r = ret.shape[-2]
            c = ret.shape[-1]
            pad_r = align_up_tile(r)
            pad_c = align_up_tile(c)
            ret = torch.nn.functional.pad(ret, (0, pad_c - c, 0, pad_r - r))
        return ret

    if type == "reshape":
        assert len(attr) >= 8 or len(attr) <= 10, "Reshape should have 8~10 attributes"
        if len(t_ops[0].shape) == 4:
            orig_w = attr[0]
            orig_z = attr[1]
            orig_r = attr[2]
            orig_c = attr[3]
            ret = t_ops[0][:,:,:orig_r,:orig_c]
        else:
            assert len(t_ops[0].shape) == 5
            orig_v = attr[0]
            orig_w = attr[1]
            orig_z = attr[2]
            orig_r = attr[3]
            orig_c = attr[4]
            ret = t_ops[0][:,:,:,:orig_r,:orig_c]

        attr_len = len(attr) - len(t_ops[0].shape)
        w = attr[-4]
        z = attr[-3]
        r = attr[-2]
        c = attr[-1]
        if attr_len == 5:
            v = attr[-5]
            ret = ret.reshape(v, w, z, r, c)
        else:
            assert attr_len == 4
            ret = ret.reshape(w, z, r, c)
        pad_r = align_up_tile(r)
        pad_c = align_up_tile(c)

        return torch.nn.functional.pad(ret, (0, pad_c - c, 0, pad_r - r))

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        assert dim == 1, f"HW only selects on t dim={dim} shape={t_ops[0].shape}"
        zero_shape = list(t_ops[0].shape)
        zero_shape[dim] = 1
        zero_slice = torch.zeros(zero_shape, dtype=dtype).squeeze(dim)
        result = []
        for offset in range(0, t_ops[0].shape[dim] - begin, stride):
            for i in range(begin, begin + length):
                if offset + i < t_ops[0].shape[dim] or stride == t_ops[0].shape[dim]:
                    result.append(t_ops[0].select(dim, offset + i))
                else:
                    result.append(zero_slice)
        return pybuda.tensor.pad_pytorch_tensor_to_buda(torch.stack(result, dim=dim), [])

    if type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, begin, length, stride, orig_size = attr
        assert dim == 1, f"HW only gathers on t dim={dim} shape={t_ops[0].shape}"
        result = []
        zero_shape = list(t_ops[0].shape)
        zero_shape[dim] = 1
        zero_slice = torch.zeros(zero_shape, dtype=dtype).squeeze(dim)
        offset = 0
        for i in range(0, orig_size):
            range_i = (i - begin) % stride
            if i >= begin and range_i < length:
                result.append(t_ops[0].select(dim, offset))
                offset += 1
            else:
                result.append(zero_slice)
        return pybuda.tensor.pad_pytorch_tensor_to_buda(torch.stack(result, dim=dim), [])

    if type == "hslice":
        assert len(attr) == 1, "HSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = t_ops[0].shape
        assert shape[3] % slice_size == 0
        ret = t_ops[0].reshape(-1, shape[2], slice_size, shape[3] // slice_size)
        ret = ret.permute(0, 2, 1, 3)
        return ret.reshape(shape[0], shape[1] * slice_size, shape[2], shape[3] // slice_size)

    if type == "hstack":
        assert len(attr) == 1, "hstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        shape = t_ops[0].shape
        assert shape[1] > 1, f"HStack requires Z to be more than 1"
        assert shape[1] % slice_size == 0, f"HStack requires Z to be divisible by slice size"
        ret = t_ops[0].reshape(-1, shape[1] // slice_size, slice_size, shape[2], shape[3])
        ret = ret.permute(0, 1, 3, 2, 4)
        return ret.reshape(shape[0], shape[1] // slice_size, shape[2], shape[3] * slice_size)

    if type == "vslice":
        assert len(attr) == 1, "VSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = t_ops[0].shape
        assert shape[2] % slice_size == 0
        return t_ops[0].reshape(shape[0], shape[1] * slice_size, shape[2] // slice_size, shape[3])

    if type == "vstack":
        assert len(attr) == 1, "vstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        shape = t_ops[0].shape
        assert shape[1] > 1, f"VStack requires Z to be more than 1"
        assert shape[1] % slice_size == 0, f"VStack requires Z to be divisible by slice size"
        return t_ops[0].reshape(shape[0], shape[1] // slice_size, shape[2] * slice_size, shape[3])

    if type == "broadcast":

        assert len(attr) <= 3, "Broadcast should have two attributes - dim and size"
        dim = attr[0]
        factor = attr[1]
        assert dim > 0, "Don't support broadcasting on w"

        if t_ops[0].is_sparse:
            return bcast_sparse_picker_matrix(t_ops[0], dim, size)

        sizes = [1] * len(t_ops[0].shape)
        sizes[dim] = factor
        return t_ops[0].repeat(*sizes)

    if type == "tile_broadcast":
        assert len(attr) == 1, "Tile broadcast should have one attribute - dim"
        dim = attr[0]
        assert t_ops[0].shape[dim] == TILE_DIM, "Tile broadcast can only be done on dim that's size of one tile"

        op = t_ops[0].narrow(dim, 0, 1)
        sizes = [1] * len(op.shape)
        sizes[dim] = 32

        return op.repeat(*sizes)

    if type == "conv2d_depthwise_weights":
        weights = t_ops[0]
        w = weights.shape[0]
        z = weights.shape[1]
        cout = weights.shape[3]
        # output_group = attr[3] // attr[0]
        return weights

    if type == "conv2d_depthwise_weights_bw":
        assert False, "Should have been const eval'd"

    if type == "conv2d_grouped_weights":
        weights = t_ops[0]
        w = weights.shape[0]
        z = weights.shape[1]
        cout = weights.shape[3]
        output_group = attr[3] // attr[0]
        weights = torch.nn.functional.pad(weights, (0, align_up_tile(cout) - cout))

        weights = weights.narrow(2, 0, attr[2])
        cin = weights.shape[2]

        weights = weights.reshape(w, z, -1, weights.shape[-1])
        weights_sections = torch.split(weights, output_group, dim=-1)
        new_weights = torch.zeros(w, z, align_up_tile(attr[0] * cin), align_up_tile(cout))
        for i, section in enumerate(weights_sections):
            new_weights[
                :,
                :,
                i*section.shape[-2]: (i+1) * section.shape[-2],
                i*section.shape[-1]: (i+1) * section.shape[-1],
            ] = section

        weights = new_weights.unsqueeze(-3)
        if len(attr) == 4:
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w, z, TILE_DIM, -1)
        elif len(attr) == 5:
            weights = weights.transpose(1, 2)
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w,1, align_up_tile(attr[0] * cin), -1)
        return weights

    if type == "conv2d_grouped_weights_bw":
        assert False, "Should have been const eval'd, comment out this line to unblock"
        weights = t_ops[0]
        w = attr[0]
        z = attr[1]
        cin = attr[2]
        cout = attr[3]
        if len(attr) == 4:
            assert weights.shape[0] == w
            assert weights.shape[1] == z
            assert weights.shape[2] == TILE_DIM
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w, z, -1, TILE_DIM, TILE_DIM)
        elif len(attr) == 5:
            weights = weights.reshape(
                w, align_up_tile(cout) // TILE_DIM, TILE_DIM, -1, TILE_DIM
            )
            weights = weights.transpose(2, 3)
            weights = weights.transpose(1, 2)
            weights = weights.reshape(w, z, -1, TILE_DIM, TILE_DIM)
        weights = weights.diagonal(dim1=-2, dim2=-1)
        weights = weights.reshape(w, z, 1, -1)[:,:,:,:cout]
        return weights

    if type == "conv2d_prestride_act":
        assert len(attr) == 6, "conv2d_prestride_act should have 6 attributes"
        stride_height, stride_width, kernel_height, kernel_width, original_y, original_x = attr

        act = t_ops[0]

        prestrided_activations = []
        for y in range(stride_height):
            for x in range(stride_width):
                prestrided_activations.append(act[:, :, y:align_up(original_y, stride_height):stride_height, x:align_up(original_x, stride_width):stride_width])

        prestrided_activations = torch.cat(prestrided_activations, dim=-3)

        w, z, r, c = prestrided_activations.shape
        prestrided_activations = prestrided_activations.view(w, 1, z, r * c)
        # prestrided_activations = prestrided_activations.transpose(-1, -2)

        # Buda padding to tile size
        r = prestrided_activations.shape[-2]
        c = prestrided_activations.shape[-1]
        pad_r = align_up_tile(r) - r
        pad_c = align_up_tile(c) - c
        prestrided_activations = torch.nn.functional.pad(prestrided_activations, (0, pad_c, 0, pad_r)) 

        return prestrided_activations

    if type == "buda_pad":
        assert len(attr) == 3, "Buda pad should have three attributes. The paddings for R and C dimensions and the value to pad with."
        r_tiles, c_tiles, value = attr
        operand = t_ops[0]
        shape = operand.shape
        # Padding is always given in tiles, so we need to recompute the padding in the original dimension
        new_r_size, new_c_size = 0, 0
        if r_tiles > 0:
            new_r_size = r_tiles * TILE_DIM
        if c_tiles > 0:
            new_c_size = c_tiles * TILE_DIM
        return torch.nn.functional.pad(operand, [0, new_c_size, 0, new_r_size], value=value)

    if type == "buda_unpad":
        assert len(attr) == 4, "Padding unpad should have four attributes. The paddings and the original shape."
        r_tiles, c_tiles, orig_r, orig_c = attr
        orig_r = align_up_tile(orig_r)
        orig_c = align_up_tile(orig_c)
        operand = t_ops[0]
        if r_tiles > 0:
            assert operand.shape[-2] == orig_r + r_tiles * TILE_DIM
        if c_tiles > 0: 
            assert operand.shape[-1] == orig_c + c_tiles * TILE_DIM
        result = torch.index_select(operand, -2, torch.arange(orig_r))
        result = torch.index_select(result, -1, torch.arange(orig_c))
        return result

    assert False, f"{type} not defined in tensor manipulations"

def shape(type, attr, ops, tile_height, tile_width):
    assert len(ops) == 1, "Tensor manipulation ops should have one input"

    if type == "transpose":
        assert len(attr) == 3
        dim0 = attr[0]
        dim1 = attr[1]
        shape = ops[0]
        a = shape[dim0]
        b = shape[dim1]
        shape[dim0] = b
        shape[dim1] = a
 
        if attr[0] < 0:
            attr[0] = attr[0] + len(shape)

        if attr[2] > 0:
            # remove padding when transpose to Z dim
            if 0 in attr[:2] and 1 not in attr[:2]:
                shape[-4] = attr[2]
            else:
                shape[-3] = attr[2]

            if attr[0] == 0:
                shape[-4] = attr[2]
            elif attr[0] == 1:
                shape[-3] = attr[2]
            else:
                assert False, "Shouldnt get to this point"
        shape[-2] = align_up_tile(shape[-2])
        shape[-1] = align_up_tile(shape[-1])
        return tuple(shape), []

    if type == "reshape":
        assert len(attr) >= 8, "Reshape should have more than 8 attributes"
        input_length = len(ops[0])
        return tuple(attr[input_length:-2] + list(map(align_up_tile, attr[-2:]))), []

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        shape = ops[0]
        if dim < 0:
            dim += max(len(shape), 4)
        shape[dim] = length * round_up_div(shape[dim] - begin, stride)
        if dim >= 2:
            shape[dim] = align_up_tile(shape[dim])
        return tuple(shape), []

    if type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, begin, length, stride, orig_size = attr
        orig_shape = ops[0]
        if dim < 0:
            dim += max(4, len(shape))
        orig_shape[dim] = orig_size
        if dim >= 2:
            orig_shape[dim] = align_up_tile(orig_shape[dim])
        return tuple(orig_shape), []

    if type == "hslice":
        assert len(attr) == 1, "HSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = ops[0]
        tile_r = align_up(shape[-2], tile_height)
        return tuple(shape[:-3] + [shape[-3] * slice_size, tile_r, shape[-1] // slice_size]), []

    if type == "hstack":
        assert len(attr) == 1, "hstack should have one attribute, equal to number of stacks of Z dim to create"
        shape = ops[0]
        slice_size = attr[0]
        assert shape[-3] > 1, f"HStack requires Z to be more than 1"
        assert shape[-3] % slice_size == 0, f"HStack requires Z to be divisible by slice size"
        tile_r = align_up(shape[-2], tile_height)
        return tuple(shape[:-3] + [shape[-3] // slice_size, tile_r, shape[-1] * slice_size]), []

    if type == "vslice":
        assert len(attr) == 1, "VSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = ops[0]
        return tuple(shape[:-3] + [shape[-3] * slice_size, shape[-2] // slice_size, shape[-1]]), []

    if type == "vstack":
        assert len(attr) == 1, "vstack should have one attribute, equal to number of stacks of Z dim to create"
        shape = ops[0]
        slice_size = attr[0]
        assert shape[-3] > 1, f"VStack requires Z to be more than 1"
        assert shape[-3] % slice_size == 0, f"VStack requires Z to be divisible by slice size"
        return tuple(shape[:-3] + [shape[-3] // slice_size, shape[-2] * slice_size, shape[-1]]), []

    if type == "broadcast":
        assert len(attr) <= 3, "Broadcast should have two attributes - dim and size"
        dim = attr[0]
        factor = attr[1]
        target_shape = ops[0]
        target_shape[-2] = align_up(target_shape[-2], tile_height)
        target_shape[-1] = align_up(target_shape[-1], tile_width)
        target_shape[dim] *= factor
        return tuple(target_shape), []

    if type == "tile_broadcast":
        # shape no-op because buda shapes already are at tile-level
        return ops[0], []

    if type == "conv2d_depthwise_weights":
        assert False, "Should have been const eval'd"  # TODO: Confirm we don't hit this

    if type == "conv2d_depthwise_weights_bw":
        assert False, "Should have been const eval'd"

    if type == "conv2d_grouped_weights":
        shape = ops[0]
        if len(attr) == 4:
            shape[2] = TILE_DIM
        elif len(attr) == 5:
            _, k, inC, outC = shape
            shape[1] = 1
            shape[2] = align_up_tile(attr[0] * attr[2])
            shape[3] = k * align_up_tile(outC)
        return tuple(shape), []

    if type == "conv2d_grouped_weights_bw":
        assert False, "Should have been const eval'd, comment out this line to unblock"
        shape = ops[0]
        if len(attr) == 4:
            assert shape[2] == TILE_DIM
            shape[2] = 1
        elif len(attr) == 5:
            w, k, cin, cout, _ = attr
            shape[1] = k
            shape[2] = cin
            shape[3] = cout
        return tuple(shape), []

    if type == "conv2d_prestride_act":
        assert len(attr) == 6, "conv2d_prestride_act should have 6 attributes"
        stride_height, stride_width, kernel_height, kernel_width, original_y, original_x = attr

        shape = ops[0]
        assert len(shape) == 4

        shape[-2] = original_y
        shape[-1] = original_x

        shape[-2] = (shape[-2] + stride_height - 1) // stride_height
        shape[-1] = (shape[-1] + stride_width - 1) // stride_width

        shape[-3] *= stride_height * stride_width

        # reshape (no transpose in Prestride transform in BE tilize)
        reshape_shape = [
            shape[0],
            1,
            shape[1],
            shape[2] * shape[3],
        ]

        # Pad to tile size
        reshape_shape[-2] = align_up_tile(shape[-2])
        reshape_shape[-1] = align_up_tile(shape[-1])

        return tuple(reshape_shape), []

    if type == "buda_pad":
        assert len(attr) == 3, "Buda pad should have three attributes. The paddings for R and C dimensions and the value to pad with."
        r_tiles, c_tiles, value = attr
        shape = ops[0]
        # Padding is always given in tiles, so we need to recompute the padding in the original dimension
        if r_tiles > 0:
            shape[-2] += r_tiles * TILE_DIM
        if c_tiles > 0:
            shape[-1] += c_tiles * TILE_DIM
        return tuple(shape), []

    if type == "buda_unpad":
        assert len(attr) == 4, "Buda unpad should have four attributes. The paddings and the original shape."
        r_tiles, c_tiles, orig_r, orig_c = attr
        orig_r = align_up_tile(orig_r)
        orig_c = align_up_tile(orig_c)
        if r_tiles > 0:
            assert ops[0][-2] == orig_r + r_tiles * TILE_DIM
        if c_tiles > 0:
            assert ops[0][-1] == orig_c + c_tiles * TILE_DIM
        shape = list(ops[0])
        shape[-2] = orig_r
        shape[-1] = orig_c
        return tuple(shape), []

    assert False, f"{type} not defined in tensor manipulations"

def parallelization(type, attr, op_shape):
    return None

def input_ublock_order(type, attr, num_operands):
    return None

def execution_cycles(type, arch_name, op_model) -> int:
    if type == "reshape":
        # TODO
        return 0
    elif type == "select":
        # TODO
        return 0
    elif type == "gather":
        # TODO
        return 0
    elif type == "conv2d_depthwise_weights":
        # TODO
        return 0
    elif type == "conv2d_grouped_weights":
        # TODO
        return 0
    else:
        assert False, "Only reshape should be a tm op in buda at this point"
    
