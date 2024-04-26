# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentError
from json.encoder import py_encode_basestring
from ssl import OP_NO_RENEGOTIATION
from ..common import to_torch_operands
from ..sparse_utils import (
    create_index_sparse_picker_matrix,
    create_all_around_padding_picker_matrix,
    create_padding_shift_sparse_picker_matrix,
    create_real_row_sparse_picker_matrix,
    create_reshape_flatten_sparse_picker_matrix,
    create_flattened_padding_removal_sparse_picker_matrix,
    create_sparse_interleave_picker_matrix,
    create_reshape_flatten_sparse_picker_matrix_narrower,
    create_repeat_sparse_picker_matrix,
    calculate_conv2d_prestride_weights_and_padding,
    create_pad_replicate_sparse_picker,
    create_pad_reflect_sparse_picker,
)
import numpy as np
import torch
import math
import ast
import os
from loguru import logger
import pybuda
from pybuda.tensor import change_rank
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile, round_up_div, align_up
from pybuda._C.balancer import FactorizedInt
from .transpose import TransposeTM
from ..buda.splice import Splice
from .nop import Nop
from ..buda.nop import Nop as BudaNop
from .buffer import Buffer

def eval(type, attr, ops):
    assert len(ops) == 1 or (type == "adv_index" and len(ops) == 2), f"Tensor manipulation ops should have one input {len(ops)} {attr}"
    t_ops = to_torch_operands(*ops)
    dtype = ops[0].dtype

    if type == "transpose":
        assert len(attr) == 3, "Transpose should have 3 attributes"
        dim0, dim1, orig_size = attr
        return torch.transpose(t_ops[0], dim0, dim1)

    if type == "reshape":
        return t_ops[0].reshape(attr)

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
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
        return torch.stack(result, dim=dim)

    if type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, begin, length, stride, orig_size = attr
        x = t_ops[0]
        result = []
        zero_shape = list(x.shape)
        if dim > 0:
            dim -= 4
        while len(zero_shape) <= abs(dim):
            zero_shape = [1] + zero_shape
            x = x.unsqueeze(0)
        zero_shape[dim] = 1
        zero_slice = torch.zeros(zero_shape, dtype=dtype).squeeze(dim)
        offset = 0
        for i in range(0, orig_size):
            range_i = (i - begin) % stride
            if i >= begin and range_i < length:
                result.append(x.select(dim, offset))
                offset += 1
            else:
                result.append(zero_slice)
        return torch.stack(result, dim=dim)

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr
        if dim >= 0:
            dim -= len(ops[0].shape)

        if dim == -5:
            return t_ops[0][..., start:stop:stride, :, :, :, :]
        elif dim == -4:
            return t_ops[0][..., start:stop:stride, :, :, :]
        elif dim == -3:
            return t_ops[0][..., start:stop:stride, :, :]
        elif dim == -2:
            return t_ops[0][..., start:stop:stride, :]
        elif dim == -1:
            return t_ops[0][..., start:stop:stride]
        else:
            raise NotImplementedError(f"Dim={dim}")

    if type == "adv_index":
        assert len(attr) == 1, "AdvIndex should have 1 attributes"
        dim = attr[0]
        assert dim == 0, "Currently not supported"

        if len(t_ops[1].shape) > 1:
            if len(t_ops[0].shape) > len(t_ops[1].shape) and t_ops[0].shape[0] == 1:
                # Padded
                ret = torch.unsqueeze(t_ops[0][0][t_ops[1].numpy()], 0)
            else:
                ret = torch.unsqueeze(t_ops[0][t_ops[1].numpy()], 0)
        else:
            ret = t_ops[0][t_ops[1].numpy()]
        return ret

    if type == "hslice":
        assert len(attr) == 1, "HSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = list(t_ops[0].shape)
        assert shape[-1] % slice_size == 0
        while len(shape) < 4:
            shape = [1] + shape
        ret = t_ops[0].reshape(-1, shape[-2], slice_size, shape[-1] // slice_size)
        ret = ret.permute(0, 2, 1, 3)
        return ret.reshape(
            shape[:-3] + [shape[-3] * slice_size, shape[-2], shape[-1] // slice_size]
        )

    if type == "hstack":
        assert (
            len(attr) == 1
        ), "hstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        shape = list(t_ops[0].shape)
        assert (
            shape[-3] % slice_size == 0
        ), f"HStack requires Z to be divisible by slice size"
        ret = t_ops[0].reshape(
            -1, shape[-3] // slice_size, slice_size, shape[-2], shape[-1]
        )
        ret = ret.permute(0, 1, 3, 2, 4)
        return ret.reshape(
            shape[:-3] + [shape[-3] // slice_size, shape[-2], shape[-1] * slice_size]
        )

    if type == "vslice":
        assert len(attr) == 1, "VSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = t_ops[0].shape
        assert len(shape) >= 2
        assert shape[-2] % slice_size == 0
        if len(shape) < 3:
            shape = (1,) + shape
        return t_ops[0].reshape(
            shape[:-3] + (shape[-3] * slice_size, shape[-2] // slice_size, shape[-1])
        )

    if type == "vstack":
        assert (
            len(attr) == 1
        ), "vstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        shape = t_ops[0].shape
        assert (
            shape[-3] % slice_size == 0
        ), f"VStack requires Z to be divisible by slice size"
        return t_ops[0].reshape(
            shape[:-3] + (shape[-3] // slice_size, shape[-2] * slice_size, shape[-1])
        )

    if type == "broadcast":
        assert len(attr) <= 3, "Broadcast should have two attributes - dim and size"
        explicit_bcast = len(attr) == 3 and bool(attr[2])

        tensor = t_ops[0]
        dim = attr[0]
        size = attr[1]
        while len(tensor.shape) <= ((-dim - 1) if dim < 0 else dim):
            tensor = tensor.unsqueeze(0)
        target_shape = list(tensor.shape)
        assert dim < len(
            target_shape
        ), f"Trying to broadcast on dim that doesn't exist: {dim} on {target_shape}"
        target_shape[dim] = size
        return torch.broadcast_to(tensor, target_shape)

    if type == "repeat":
        sizes = attr
        assert len(t_ops[0].shape) == len(sizes)
        return t_ops[0].repeat(*sizes)

    if type == "repeat_dim":
        assert len(attr) <= 3, "repeat should have two attributes - dim and size"
        dim = attr[0]
        if dim < 0:
            dim += len(t_ops[0].shape)
        factor = attr[1]
        assert dim > 0, "Don't support broadcasting on w"
        sizes = [1] * len(t_ops[0].shape)
        sizes[dim] = factor
        return t_ops[0].repeat(*sizes)

    if type == "conv2d_depthwise_weights":
        weights = t_ops[0]
        assert len(weights.shape) == 4, "Weights should have rank 4"

        w, z, cin, cout = weights.shape

        assert cin == 1, "Depthwise weights should always have cin == 1"

        # [1, 9, 1, 65] -> [1, 9, 1, 96]
        weights = torch.nn.functional.pad(weights, (0, align_up_tile(cout) - cout))
        # [1, 9, 1, 96] -> [1, 9, 32, 96]
        weights = torch.nn.functional.pad(weights, (0, 0, 0, align_up_tile(cin) - cin))

        # Diagonally embed weights
        weights_diag = torch.zeros_like(weights, requires_grad=False)

        cnt_kernels = z
        ct = weights.shape[-1] // TILE_DIM
        for idx_kernel in range(cnt_kernels):
            for idx_ct in range(ct):
                weights_diag[:, idx_kernel, :, idx_ct * TILE_DIM: (idx_ct + 1) * TILE_DIM] = \
                    torch.diag_embed(weights[:, idx_kernel, 0, idx_ct * TILE_DIM: (idx_ct + 1) * TILE_DIM])

        # [1, 9, 32, 96] -> [1, 1, 9 * 32, 96]
        weights_diag = weights_diag.reshape(w, 1, -1, weights.shape[-1])

        return weights_diag

    if type == "conv2d_depthwise_weights_bw":
        assert False, "not implemented yet"

    if type == "conv2d_grouped_weights":
        weights = t_ops[0]
        w = weights.shape[0]
        z = weights.shape[1]
        cin = weights.shape[2]
        cout = weights.shape[3]
        output_group = cout // attr[0]

        weights = torch.nn.functional.pad(weights, (0, align_up_tile(cout) - cout))
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
        weights = t_ops[0]
        groups = attr[0]
        w = 1
        z = attr[1]
        cin = attr[2]
        cout = attr[3]
        output_group = cout // groups

        if len(attr) == 4:
            assert weights.shape[0] == w
            assert weights.shape[1] == z
            assert weights.shape[2] == TILE_DIM
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w, z, -1, TILE_DIM, TILE_DIM)
        elif len(attr) == 5:
            weights = weights.reshape(w,1, align_up_tile(groups * cin), -1)
            weights = weights.transpose(2, 3)
            weights = weights.transpose(1, 2)
            weights = weights.reshape(w, z, align_up_tile(groups * cin), align_up_tile(cout))

        sections = []
        for i in range(groups):
            section = weights[
                :,:,i * cin : (i + 1) * cin, i * output_group : (i + 1) * output_group
            ]
            sections.append(section)

        new_weights = torch.concat(sections,dim=-1)

        weights = new_weights.reshape(w, z, cin, -1)[:, :, :, :cout]
        return weights

    if type == "conv2d_prestride_act":
        assert len(attr) == 6, "conv2d_prestride_act should have 6 attributes"
        stride_height, stride_width, kernel_height, kernel_width, original_y, original_x = attr

        act = t_ops[0]

        act = torch.nn.functional.pad(act, (0, align_up(original_x, stride_width) - original_x, 0, align_up(original_y, stride_height) - original_y)) 

        prestrided_activations = []
        for y in range(stride_height):
            for x in range(stride_width):
                prestrided_activations.append(act[:, :, y::stride_height, x::stride_width])

        prestrided_activations = torch.cat(prestrided_activations, dim=-3)

        w, z, r, c = prestrided_activations.shape
        prestrided_activations = prestrided_activations.view(w, 1, z, r * c)
        # prestrided_activations = prestrided_activations.transpose(-1, -2)

        return prestrided_activations

    if type == "conv2d_prestride_weights":
        assert len(attr) == 8, "conv2d_prestride_weights should have 8 attributes"
        y, x = attr[0], attr[1]
        stride_height, stride_width = attr[2], attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7]]

        weights = t_ops[0]
        assert len(weights.shape) == 4, "weights should have 4 dims"

        ps_weights, _ = calculate_conv2d_prestride_weights_and_padding(weights, y, x, stride_width, padding)
        return ps_weights

    if type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        act = t_ops[0]
        if dim >= 0:
            dim -= len(act.shape)
        assert dim == -2 or dim == -1
        padding = align_up_tile(act.shape[dim]) - act.shape[dim]
        if dim == -2:
            act = torch.nn.functional.pad(act, (0, 0, 0, padding))
        if dim == -1:
            act = torch.nn.functional.pad(act, (0, padding))
        return act

    if type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        act = t_ops[0]
        return act.narrow(dim, start, length)

    if type == "pad":
        # Expect (padding_left, padding_right, mode, channel_last) or (padding_left, padding_right, padding_top, padding_bottom, mode, channel_last)
        assert len(attr) == 4 or len(attr) == 6
       
        mode_idx = attr[-2]
        channel_last = attr[-1]
        attr = attr[:-2]
        if channel_last:
            attr = [0, 0] + attr
        mode_options = ["constant", "replicate", "reflect"]
        return torch.nn.functional.pad(t_ops[0], tuple(attr), mode=mode_options[mode_idx])

    if type == "unsqueeze":
        assert len(attr) == 2
        dim = attr[0]
        input_ndim = attr[1]
        act = t_ops[0]
        return torch.unsqueeze(act, dim)

    if type == "squeeze":
        assert len(attr) == 1
        dim = attr[0]
        act = t_ops[0]
        return torch.squeeze(act, dim)
    
    if type == "pixel_shuffle":
        assert len(ops) == 1, "Pixel shuffle should have one operand."
        assert len(attr) == 1, "Pixel shuffle should have one attribute."
        return torch.nn.functional.pixel_shuffle(ops[0], attr[0])

    if type == "buda_pad":
        assert len(attr) == 3, "Buda pad should have three attributes. The paddings for R and C dimensions and the value to pad with."
        r_tiles, c_tiles, value = attr
        operand = t_ops[0]
        shape = operand.shape
        # Padding is always given in tiles, so we need to recompute the padding in the original dimension
        new_r_size_tile, new_c_size_tile = 0, 0
        new_r_size, new_c_size = 0, 0
        if r_tiles > 0:
            new_r_size_tile = align_up_tile(shape[-2]) - shape[-2] 
            new_r_size = r_tiles * TILE_DIM
        if c_tiles > 0:
            new_c_size_tile = align_up_tile(shape[-1]) - shape[-1]
            new_c_size = c_tiles * TILE_DIM
        result = torch.nn.functional.pad(operand, [0, new_c_size_tile, 0, new_r_size_tile], value=0)
        result = torch.nn.functional.pad(result, [0, new_c_size, 0, new_r_size], value=value)
        return result

    if type == "buda_unpad":
        assert len(attr) == 4, "Buda unpad should have four attributes. The paddings and the original shape."
        r_tiles, c_tiles, orig_r, orig_c = attr
        operand = t_ops[0]
        if r_tiles > 0:
            assert operand.shape[-2] == align_up_tile(orig_r) + r_tiles * TILE_DIM
        if c_tiles > 0:
            assert operand.shape[-1] == align_up_tile(orig_c) + c_tiles * TILE_DIM
        result = torch.index_select(operand, -2, torch.arange(orig_r))
        result = torch.index_select(result, -1, torch.arange(orig_c))
        return result

    assert False, f"{type} not defined in tensor manipulations"


def shape(type, attr, ops):
    assert len(ops) == 1 or (type == "adv_index" and len(ops) == 2), f"Tensor manipulation ops should have one input, has {len(ops)} input instead"

    if type == "transpose":
        # Transpose has 3 attrs, [axis_0, axis_1, output Z-dim size]
        assert len(attr) == 3, f"{len(attr)}"
        dim0 = attr[0]
        dim1 = attr[1]
        shape = list(ops[0])
        a = shape[dim0]
        b = shape[dim1]
        shape[dim0] = b
        shape[dim1] = a
        return tuple(shape), []

    if type == "reshape":
        return attr, []

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr
        shape = list(ops[0])
        
        if start < 0:
            start = shape[dim] + start
         
        shape[dim] = round_up_div(stop - start, stride)
        return tuple(shape), []

    if type == "adv_index":
        assert len(attr) == 1, "AdvIndex should have 1 attributes"
        dim = attr[0]
        assert dim == 0, "Currently not supported"
        shape = list(ops[0])
        shape[dim] = ops[1][-1]
        if len(ops[1]) > 1:
            shape.insert(dim, 1)
        return shape, []

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        shape = list(ops[0])
        shape[dim] = length * round_up_div(shape[dim] - begin, stride)
        return tuple(shape), []

    if type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, begin, length, stride, orig_size = attr
        orig_shape = list(ops[0])
        if dim > 0:
            dim -= 4
        while len(orig_shape) <= abs(dim):
            orig_shape = [1] + orig_shape
        orig_shape[dim] = orig_size
        return tuple(orig_shape), []

    if type == "hslice":
        assert len(attr) == 1, "HSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = list(ops[0])
        assert shape[-1] % slice_size == 0
        while len(shape) < 4:
            shape = [1] + shape
        shape[-1] //= slice_size
        shape[-3] *= slice_size
        return tuple(shape), []

    if type == "hstack":
        assert len(ops[0]) >= 3, "HStack should at least have 3 dims"
        assert (
            len(attr) == 1
        ), "hstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        assert (
            ops[0][-3] % slice_size == 0
        ), f"HStack requires Z: {ops[0][-3]} to be a multiple of slice_size: {slice_size},"
        shape = list(ops[0])
        shape[-1] *= slice_size
        shape[-3] //= slice_size
        return tuple(shape), []

    if type == "vslice":
        assert len(attr) == 1, "VSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = list(ops[0])
        assert len(shape) >= 2, "VSlice should at least have 2 dims"
        assert shape[-2] % slice_size == 0
        while len(shape) < 3:
            shape = [1] + shape
        shape[-2] //= slice_size
        shape[-3] *= slice_size
        return tuple(shape), []

    if type == "vstack":
        assert len(ops[0]) >= 3, "VStack should at least have 3 dims"
        assert (
            len(attr) == 1
        ), "vstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        assert (
            ops[0][-3] % slice_size == 0
        ), f"VStack requires Z to be a multiple of slice_size"
        shape = list(ops[0])
        shape[-2] *= slice_size
        shape[-3] //= slice_size
        return tuple(shape), []

    if type == "broadcast":
        assert len(attr) <= 3, "Broadcast should have two attributes - dim and size"
        dim = attr[0]
        size = attr[1]
        target_shape = list(ops[0])
        
        if dim < 0:
            while abs(dim) > len(target_shape):
                target_shape = [1] + target_shape
        else:
            while dim >= len(target_shape):
                target_shape = [1] + target_shape

        target_shape[dim] = size
        return tuple(target_shape), []

    if type == "repeat":
        sizes = attr
        return tuple(dim * size for dim, size in zip(list(ops[0]), sizes)), []

    if type == "repeat_dim":
        assert len(attr) <= 3, "repeat should have two attributes - dim and size"
        dim = attr[0]
        if dim < 0:
            dim += len(ops[0])
        factor = attr[1]
        target_shape = list(ops[0])
        target_shape[dim] *= factor
        return tuple(target_shape), []

    if type == "conv2d_depthwise_weights":
        shape = list(ops[0])

        w, k, _, cout = shape
        shape = [w, 1, k * TILE_DIM, align_up_tile(cout)]

        return tuple(shape), []

    if type == "conv2d_depthwise_weights_bw":
        assert False, "not yet implemented"

    if type == "conv2d_grouped_weights":
        shape = list(ops[0])
        if len(attr) == 4:
            shape[2] = TILE_DIM
        elif len(attr) == 5:
            _, k, cin, cout = shape
            shape[1] = 1
            shape[2] = align_up_tile(attr[0] * cin)
            shape[3] = k * align_up_tile(cout)
        return tuple(shape), []

    if type == "conv2d_grouped_weights_bw":
        shape = list(ops[0])
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

        shape = list(ops[0])
        assert len(shape) == 4

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

        return tuple(reshape_shape), []


    if type == "conv2d_prestride_weights":
        assert len(attr) == 8, "conv2d_prestride_weights should have 8 attributes"
        y, x = attr[0], attr[1]
        stride_height, stride_width = attr[2], attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7]]

        shape = list(ops[0])
        assert len(shape) == 4
        shape, _ = calculate_conv2d_prestride_weights_and_padding(shape, y, x, stride_width, padding)
        return tuple(shape), []

    if type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        if dim >= 0:
            dim -= len(ops[0])
        if not (dim == -2 or dim == -1):
            x =2
        assert dim == -2 or dim == -1
        shape = list(ops[0])
        shape[dim] = align_up_tile(shape[dim])
        return tuple(shape), []

    if type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        shape = list(ops[0])
        shape[dim] = length
        return tuple(shape), []

    if type == "pad":
        assert len(attr) == 4 or len(attr) == 6
        shape = list(ops[0])
        channel_last = attr[-1]

        if channel_last:
            shape[-2] += attr[0] + attr[1]
            if len(attr) == 6:
                shape[-3] += attr[2] + attr[3]
        else:
            shape[-1] += attr[0] + attr[1]
            if len(attr) == 6:
                shape[-2] += attr[2] + attr[3]
        return tuple(shape), []

    if type == "unsqueeze":
        assert len(attr) == 2
        shape = list(ops[0])
        dim = attr[0]
        input_ndim = attr[1]
        shape.insert(dim, 1)
        return tuple(shape), []

    if type == "squeeze":
        assert len(attr) == 1
        shape = list(ops[0])
        dim = attr[0]
        del shape[dim]
        return tuple(shape), []
        
    if type == "pixel_shuffle":
        assert len(ops) == 1, "Pixel shuffle should have one operand."
        assert len(attr) == 1, "Pixel shuffle should have one attribute."
        
        orig_shape = ops[0]
        assert len(orig_shape) >= 3, "Pixel shuffle should be at least 3D."
        
        upscale_factor = attr[0]
        assert orig_shape[-3] % (upscale_factor**2) == 0, f"Op shape at dim -3 ({orig_shape[-3]}) should be divisible by upscale_factor*upscale_factor ({upscale_factor**2})."
        
        output_shape = (*orig_shape[:-3], orig_shape[-3] // upscale_factor**2, orig_shape[-2] * upscale_factor, orig_shape[-1] * upscale_factor)
        return output_shape, []

    if type == "buda_pad":
        assert len(attr) == 3, "Buda pad should have three attributes. The paddings for R and C dimensions and the value to pad with."
        r_tiles, c_tiles, value = attr
        shape = list(ops[0])
        # Padding is always given in tiles, so we need to recompute the padding in the original dimension
        if r_tiles > 0:
            shape[-2] = (align_up_tile(shape[-2]) // TILE_DIM + r_tiles) * TILE_DIM
        if c_tiles > 0:
            shape[-1] = (align_up_tile(shape[-1]) // TILE_DIM + c_tiles) * TILE_DIM
        return tuple(shape), []

    if type == "buda_unpad":
        assert len(attr) == 4, "Buda unpad should have four attributes. The paddings and the original shape."
        r_tiles, c_tiles, orig_r, orig_c = attr
        if r_tiles > 0:
            assert ops[0][-2] == align_up_tile(orig_r) + r_tiles * TILE_DIM
        if c_tiles > 0:
            assert ops[0][-1] == align_up_tile(orig_c) + c_tiles * TILE_DIM
        shape = list(ops[0])
        shape[-2] = orig_r
        shape[-1] = orig_c
        return tuple(shape), []

    assert False, f"{type} not defined in tensor manipulations"


def lower(type, attr, lc, ops, outputs):
    assert len(ops) == 1, "Tensor manipulation ops should have one input"

    if type == "reshape":
        while len(attr) > 4:
            assert attr[0] == 1, "Cannot eliminate non-singleton dimension"
            attr = attr[1:]
        while len(attr) < 4:
            attr.insert(0, 1)

        # Pad shape to 4D before lowering
        orig_shape = []
        for i in range(ops[0].shape.len()):
            orig_shape.append(ops[0].shape[i])
        while len(orig_shape) < 4:
            orig_shape.insert(0, 1)

        assert len(attr) == 4, "Reshape should have 4 attributes"

        # Squeeze / unsqueeze ops that do not reshape a 4d tensor are nops
        if all([orig == new for orig, new in zip(orig_shape, attr)]):
            lc.op(BudaNop.create(), ops)
        else:
            orig_w = orig_shape[-4]
            orig_z = orig_shape[-3]
            orig_r = orig_shape[-2]
            orig_c = orig_shape[-1]
            buda_attrs = {
                "orig_w": orig_w,
                "orig_z": orig_z,
                "orig_r": orig_r,
                "orig_c": orig_c,
                "w": attr[0],
                "z": attr[1],
                "r": attr[2],
                "c": attr[3],
            }
            lc.op(type, ops, (orig_w, orig_z, orig_r, orig_c, *attr), buda_attrs)

    elif type == "transpose":
        # Transpose has 3 attrs, [axis_0, axis_1, output Z-dim size]
        assert len(attr) == 3, "Transpose should have 3 attributes"
        if attr[0] < 0:
            attr[0] += ops[0].shape.len()
        if attr[1] < 0:
            attr[1] += ops[0].shape.len()

        # Adjust the broadcast dim if we're moving to more/less dimensions
        delta = 4 - ops[0].shape.len()
        attr[0] += delta
        attr[1] += delta
        assert attr[0] >= 0 and attr[0] <= 3, f"Invalid transpose dim after lowering: {attr[0]}"
        assert attr[1] >= 0 and attr[1] <= 3, f"Invalid transpose dim after lowering: {attr[0]}"

        if attr[0] == 2 and attr[1] == 3:
            lc.tm("transpose", ops[0], attr, named_attrs={"dim0": attr[0], "dim1": attr[1], "z_dim_slice": attr[2]})
        else:
            lc.op("transpose", ops, attr, {"dim0": attr[0], "dim1": attr[1], "z_dim_slice": attr[2]})

    elif type == "broadcast":
        if attr[0] < 0:
            attr[0] += ops[0].shape.len()
        # Adjust the broadcast dim if we're moving to more/less dimensions
        delta = 4 - ops[0].shape.len()
        attr[0] += delta
        assert (
            attr[0] >= 0 and attr[0] <= 3
        ), f"Invalid broadcast dim after lowering: {attr[0]}"

        if attr[0] == 2 or attr[0] == 3:
            # Adjust broadcast size if not divisible by tile dim
            attr[1] = int(math.ceil(attr[1] / TILE_DIM)) * TILE_DIM
            attr[1] //= TILE_DIM

        return lc.tm("broadcast", ops[0], attr)

    elif type == "repeat":
        assert False, "repeat should have been decomposed into repeat_dim"

    elif type == "repeat_dim":
        # Adjust the repeat dim if we're moving to more/less dimensions
        if attr[0] < 0:
            attr[0] += ops[0].shape.len()

        delta = 4 - ops[0].shape.len()
        attr[0] += delta
        assert (
            attr[0] >= 0 and attr[0] <= 3
        ), f"Invalid repeat dim after lowering: {attr[0]}"

        if attr[0] == 2:
            assert ops[0].shape[-2] % TILE_DIM == 0, "Repeat on R must be TILE_DIM aligned"
        if attr[0] == 3:
            assert ops[0].shape[-1] % TILE_DIM == 0, "Repeat on C must be TILE_DIM aligned"
        return lc.tm("broadcast", ops[0], attr)

    elif type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, index, length, stride = attr
        return lc.op(Splice.create_select(dim, index, length, stride, ops[0].shape), ops)

    elif type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, index, length, stride, orig_size = attr
        if dim >= 0:
            dim += 4 - ops[0].shape.len()
        else:
            dim += 4
        return lc.op(
            "gather",
            ops,
            (dim, index, length, stride, orig_size),
            {"index": index, "length": length, "stride": stride, "size": orig_size},
        )

    elif type == "pad_tile":
        return lc.op(BudaNop.create(), ops)

    elif type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        if dim >= 0:
            dim -= len(ops[0].shape)
        if dim >= -2 and align_up_tile(length) == align_up_tile(ops[0].shape[dim]):
            return lc.op(BudaNop.create(), ops)
        else:
            raise NotImplementedError("Unimplemented narrow in buda")

    elif type == "pad":
        assert ((len(attr) == 4 and attr[0] == 0) or 
               (len(attr) == 6 and attr[0] == 0 and attr[2] == 0) or
               (attr[-2] != 0)), "Nop does not support left/top padding for constant mode"
        return lc.op(BudaNop.create(), ops)

    elif type == "unsqueeze":
        assert len(attr) == 2
        input_ndim = attr[1]
        #assert input_ndim + 1 <= 4, "Cannot unsqueeze beyond 4D"
        if input_ndim + 1 > 4:
            assert attr[0] == 0, f"Unsqueeze 4D tensors to 5D is only supported for the 1st dim: {attr[0]}" 
            return lc.op(BudaNop.create(unsqueeze = "unsqueeze", unsqueeze_dim=attr[1]), ops, tag="dont_remove")

        return lc.op(BudaNop.create(), ops)

    elif type == "squeeze":
        assert len(attr) == 1
        if len(ops[0].shape) >= 5:
            assert attr[0] == 0, f"Squeeze 5D tensors to 4D is only supported for the 1st dim: {attr[0]}" 
            return lc.op(BudaNop.create(squeeze="squeeze", squeeze_dim=attr[0]), ops, tag="dont_remove")

        return lc.op(BudaNop.create(), ops)

    elif (type == "hstack" or type == "hslice") and attr[0] == 1:
        return lc.op(BudaNop.create(), ops)

    elif type == "buda_pad":
        return lc.tm("buda_pad", ops[0], attr, { "rt": attr[0], "ct": attr[1], "pad_value": attr[2]})

    elif type == "buda_unpad":
        return lc.tm("buda_unpad", ops[0], attr, { "rt": attr[0], "ct": attr[1], "orig_r": attr[2], "orig_c": attr[3]})

    else:
        lc.tm(type, ops[0], attr)  # straight 1-1 for other tms


def backward(type, attr, ac, operand, inputs, output, grad):

    assert operand == 0, "Invalid operand index"

    if type == "hstack":
        return ac.op("hslice", (grad,), attributes=attr)

    elif type == "hslice":
        return ac.op("hstack", (grad,), attributes=attr)

    elif type == "vstack":
        return ac.op("vslice", (grad,), attributes=attr)

    elif type == "vslice":
        return ac.op("vstack", (grad,), attributes=attr)
    
    elif type == "transpose":
        assert len(attr) == 3

        if (attr[0] == -3 and attr[1] == -4) or (attr[0] == -4 and attr[1] == -3):
            attr[-1] = -1
        elif attr[0] == -3 or attr[0] == -4:
            attr[-1] = grad.shape[attr[1]]
        elif attr[1] == -3 or attr[1] == -4:
            attr[-1] = grad.shape[attr[0]] 
        else:
            attr[-1] = -1

        return ac.op("transpose", (grad, ), attr)

    elif type == "reshape":
        shape = inputs[0].shape
        return ac.op(type, (grad,), attributes=(shape))

    elif type == "conv2d_depthwise_weights":
        return ac.op("conv2d_depthwise_weights_bw", (grad,), attributes=attr)

    elif type == "conv2d_grouped_weights":
        return ac.op("conv2d_grouped_weights_bw", (grad,), attributes=attr)

    elif type == "select":
        assert len(attr) == 4
        dim, begin, length, stride = attr
        orig_size = inputs[0].shape[dim]
        current_size = grad.shape[dim]
        #return ac.op("gather", (grad,), attributes=(dim, begin, length, stride, orig_size))

        # temporarily replace gather op (not HW supported) with select + concat   
        grad_return = None  
        grad_offset = 0     

        for offset in range(0, orig_size, stride):
            # zero padding of front
            if begin > 0:
                zero_pre_pad_shape = inputs[0].shape.as_list() 
                zero_pre_pad_shape[dim] = min(begin, orig_size-offset) 
                if grad_return is None:
                    grad_return = ac.tensor(torch.zeros(zero_pre_pad_shape))
                else:
                    zero_slice = ac.tensor(torch.zeros(zero_pre_pad_shape))
                    grad_return = ac.op("concatenate", (grad_return, zero_slice), (dim,)) 
            if offset + begin >= orig_size:
                break

            # pass the gradient for selected part
            grad_slice = ac.op("select", (grad,), (dim, grad_offset, length, current_size))
            if grad_return is None:
                grad_return = grad_slice
            else:
                grad_return = ac.op("concatenate", (grad_return, grad_slice), (dim,))
            grad_offset += length 
            if offset + begin + length >= orig_size:
                break

            # zero padding of back
            zero_padding_length = stride - length - begin
            if zero_padding_length > 0:
                zero_post_pad_shape = inputs[0].shape.as_list() 
                zero_post_pad_shape[dim] = zero_padding_length 
                zero_slice = ac.tensor(torch.zeros(zero_post_pad_shape))
                grad_return = ac.op("concatenate", (grad_return, zero_slice), (dim,)) 
        return grad_return  
        
    elif type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        return ac.op(
            "narrow",
            (grad,),
            attributes=(dim, 0, inputs[0].shape[dim], original_length),
        )

    elif type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        if dim >= 0:
            dim -= len(inputs[0].shape)
        if dim in [-1, -2] and align_up_tile(length) == align_up_tile(inputs[0].shape[dim]):
            if dim == -1:
                return ac.op("pad", (grad,), (start, original_length - length - start, 0, False))
            elif dim == -2:
                return ac.op(
                    "pad", (grad,), (0, 0, start, original_length - length - start, 0, False)
                )
            raise ArgumentError("Only dim == 2 and dim == 3 are supported.")
        else:
            raise NotImplementedError("Unimplemented narrow in buda")

    elif type == "pad":  # TODO: update it for replicate mode
        assert len(attr) == 4 or len(attr) == 6, "Not supported padding type"
        if len(attr) == 6:
            pad_left, pad_right, pad_top, pad_bottom, _, _ = attr
            original_heigth = grad.shape[-2] # input heigth
            original_width = grad.shape[-1] # input width
            grad = ac.op("narrow", (grad,), (-2, pad_top, original_heigth - pad_top - pad_bottom, original_heigth))
            return ac.op("narrow", (grad,), (-1, pad_left, original_width - pad_left - pad_right, original_width))
        elif len(attr) == 4:
            pad_left, pad_right, _, _ = attr
            original_width = grad.shape[-1] # input width
            return ac.op("narrow", (grad,), (-1, pad_left, original_width - pad_left - pad_right, original_width))

    elif type == "unsqueeze":
        assert len(attr) == 2
        if len(inputs[0].shape) == len(grad.shape):
            # Dimensionality already matches, no need to squeeze
            return grad

        dim = attr[0]
        input_ndim = attr[1]
        return ac.op("squeeze", (grad,), attributes=(dim,))

    elif type == "squeeze":
        assert len(attr) == 1
        if len(inputs[0].shape) == len(grad.shape):
            # Dimensionality already matches, no need to unsqueeze
            return grad

        dim = attr[0]
        if grad.shape.len() == 4: # Cannot unsqueeze beyond 4D
            return ac.op(Nop.create(), (grad,))
        return ac.op("unsqueeze", (grad,), attributes=(dim, grad.shape.len()))

    elif type == "broadcast": 
        assert len(attr) == 3
        if attr[0] < 0:
            attr[0] += inputs[0].shape.len()
        delta = 4 - inputs[0].shape.len()
        attr[0] += delta
        assert (
            attr[0] >= 0 and attr[0] <= 3
        ), f"Invalid broadcast dim after lowering: {attr[0]}"
 
        if attr[0] == 2 or attr[0] == 3:
             ret = ac.op("reduce_sum", (grad,), (attr[0],))
        else:
            ret = ac.op(TransposeTM.create(attr[0], -2, z_dim_slice=grad.shape[-2]), [grad,])
            ret = ac.op("reduce_sum", (ret,), (-2,))
            ret = ac.op(TransposeTM.create(attr[0], -2, z_dim_slice=ret.shape[-2]), [ret,])
        return ret 

    raise NotImplementedError(f"{type}")


def unsqueeze_input_for_reshape_decomp(dc, inp):

    current_shape = inp.shape.as_list()
    while len(current_shape) < 4:
        current_shape.insert(0, 1)
        inp = dc.op("unsqueeze", (inp,), (0, len(inp.shape.as_list())))

    return inp

def squeeze_output_for_reshape_decomp(dc, output, orig_out_shape):
    current_shape_len = 4
    assert current_shape_len == output.shape.len()

    while current_shape_len > len(orig_out_shape):
        current_shape_len -= 1
        output = dc.op("squeeze", (output,), (0,))

    return output

def decompose(type, attr, dc, inputs):
    act = inputs[0]

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr
        
        if start < 0:
            start = act.shape[dim] + start
        
        length = stop - start
        assert dim != -4, "No support for indexing on w"
        is_z_dim = dim == -3
        is_c_dim = dim == -1
        is_one_dim = len(act.shape) == 1

        if is_z_dim and stride == 1:
            result = dc.op("select", [act], (dim, start, length, act.shape[dim]))
            dc.fuse(result)
            return
        elif is_z_dim and stride > 1:
            result = dc.op("select", [act], (dim, start, 1, stride))
            if result.shape[dim] >= length:
                result = dc.op(
                    "select",
                    [result],
                    (dim, 0, round_up_div(length, stride), result.shape[dim]),
                )
            dc.fuse(result)
            return
        elif start % TILE_DIM == 0 and stop % TILE_DIM == 0 and stride == 1 and act.shape[dim] % TILE_DIM == 0:
            result = dc.op("select", [act], (dim, start, length, act.shape[dim]))
            result = dc.op(Buffer.create(), [result]) # Workaround to enable T-streaming for Splice
            dc.fuse(result)
            return
        elif act.shape[dim] == 1 and length == 1 and stride == 1:
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)
            return
        elif dim == -2 and stride == 1 and length == stop and "PYBUDA_PAD_MM" in os.environ:
            sparse_r_padding = ast.literal_eval(os.environ.get('PYBUDA_PAD_MM', "{}"))
            sparse_r = align_up_tile(attr[-2]) // 32
            if sparse_r in sparse_r_padding:
                padded_r = sparse_r_padding[sparse_r] - sparse_r
                result = dc.op("buda_unpad", [act], (padded_r, 0, attr[-2], act.shape[-1]))
                dc.fuse(result)
                return            

        result = act

        # if the non-indexed dimension is very large, stack so it can be streamed easier
        slice_amount = None
        if is_c_dim and len(inputs[0].shape) > 2 and inputs[0].shape[-2] > 4096 and inputs[0].shape[-2] % TILE_DIM == 0:
            slice_amount = (inputs[0].shape[-2] // TILE_DIM)
            result = dc.op("vslice", [result], (slice_amount, ))

        if is_one_dim:
            result = dc.op("reshape", [result], (1, result.shape[0]))
        if is_c_dim:
            result = dc.op(TransposeTM.create(-2, -1), [result])

        result_shape = result.shape
        # Fold W dim into Z
        if len(result_shape) == 4 and result_shape[0] != 1:
            result = dc.op("reshape", [result], (1, result_shape[0] * result_shape[1], result_shape[2], result_shape[3]))

        spm = create_index_sparse_picker_matrix(result.shape[-2], start, stop, stride)
        if len(result.shape) >= 3:
            spm = torch.unsqueeze(spm, 0)
            spm = torch.stack([spm] * result.shape[-3], -3)
        lhs = dc.tensor(spm)
        result = dc.op("sparse_matmul", [lhs, result])
        
        # Fold back W dim
        if len(result_shape) == 4 and result_shape[0] != 1:
            new_shape = result_shape.as_list()
            new_shape[-2] = (stop - start) // stride
            result = dc.op("reshape", [result], new_shape)

        if is_c_dim:
            result = dc.op(TransposeTM.create(-2, -1), [result])

        if slice_amount is not None:
            result = dc.op("vstack", [result], (slice_amount, ))

        if is_one_dim:
            result = dc.op("reshape", [result], (result.shape[-1],))
        dc.fuse(result)
        return

    if type == "adv_index":
        dim = attr[0]
        in0_shape = inputs[0].shape
        in1_shape = inputs[1].shape
        if len(in0_shape) == 1 or in0_shape[dim] == 1:
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)
            return
        if dim == 0 and len(in1_shape) <= 2:
            # Consider the case adv_index(X,Y) where
            #    X: (A, B), Y: (1, C) or (C,) and A != 1
            if len(in0_shape) == 2:
                result = dc.op("embedding", inputs,)
                dc.fuse(result)
                return

    if type == "pad":
        if all([x == 0 for x in attr[0:-2]]):
            # Pad size is 0
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)


        activations = inputs[0]
        mode_idx = attr[-2]
        channel_last = attr[-1]
        if channel_last:
            r = activations.shape[-3]
            c = activations.shape[-2]
        else:
            r = activations.shape[-2]
            c = activations.shape[-1]

        # Find out if padding exceeds tile boundary
        # R, C are flipped because pytorch pad starts from last axis
        if len(attr) == 4:
            total_padding_c = attr[0] + attr[1]
            total_padding_r = 0
            all_around_padding = attr[:-2] + [0, 0]
        elif len(attr) == 6:
            total_padding_c = attr[0] + attr[1]
            total_padding_r = attr[2] + attr[3]
            all_around_padding = attr[:-2]
        else:
            raise RuntimeError("Pybuda only support Pad with either 2 or 4 attributes")

        if (((len(attr) == 4 and attr[0] == 0) or 
             (len(attr) == 6 and attr[0] == 0 and attr[2] == 0)) and
            not channel_last and
            math.ceil((total_padding_r + r) / TILE_DIM) == math.ceil(r / TILE_DIM) and
            math.ceil((total_padding_c + c) / TILE_DIM) == math.ceil(c / TILE_DIM) and
            mode_idx == 0 # 'constant' mode
        ):
            # Pad does not exceed tile boundary and only on the end of axis
            # Will be lowered into NOP
            return

        else:
            # Lower into concats
            left, right, top, bottom = 0,0,0,0
            if len(attr) == 4:
                left, right, _, _ = attr

            elif len(attr) == 6:
                left, right, top, bottom, _, _ = attr
            else:
                raise RuntimeError("Pybuda only support Pad with either 3 or 5 attributes")

            if mode_idx == 1: # 'replicate' mode
                result = activations
 
                if channel_last:
                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
 
                    orig_shape = result.shape
                    result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2]*orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_replicate_sparse_picker(c, r, top, bottom, left, right)
                    spm = dc.tensor(spm)
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    result = dc.op("reshape", [result], (1, orig_shape[-3], orig_shape[-1]+total_padding_r, orig_shape[-2]+total_padding_c))

                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
                else:
                    orig_shape = result.shape
                    if len(orig_shape) == 2:
                        result = dc.op("reshape", [result], (1, orig_shape[-2]*orig_shape[-1]))
                    else:
                        result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2]*orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result]) 
                    spm = create_pad_replicate_sparse_picker(r, c, left, right, top, bottom)
                    spm = dc.tensor(spm) 
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result]) 
                    if len(orig_shape) == 2:
                        result = dc.op("reshape", [result], (orig_shape[-2]+total_padding_r, orig_shape[-1]+total_padding_c))
                    else:
                        result = dc.op("reshape", [result], (1, orig_shape[-3], orig_shape[-2]+total_padding_r, orig_shape[-1]+total_padding_c))

                dc.fuse(result)
                return

            elif mode_idx == 0: # 'constant' mode
                c_dim_axis = -2 if channel_last else -1
                r_dim_axis = -3 if channel_last else -2

                # On right or bottom, we can concat all the way to TILE boundary
                result = activations
                if left > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[c_dim_axis] = left
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [const_tensor, result], [c_dim_axis])

                if right > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[c_dim_axis] = TILE_DIM if pad_shape[c_dim_axis] % TILE_DIM == 0 and right < TILE_DIM else right
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [result, const_tensor], [c_dim_axis])

                if top > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[r_dim_axis] = top
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [const_tensor, result], [r_dim_axis])

                if bottom > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[r_dim_axis] = TILE_DIM if pad_shape[r_dim_axis] % TILE_DIM == 0 and bottom < TILE_DIM else bottom
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [result, const_tensor], [r_dim_axis])

                result = dc.op("narrow", [result], (c_dim_axis, 0, total_padding_c + c, result.shape[c_dim_axis]))
                if channel_last:
                    result = dc.op("select", [result], (r_dim_axis, 0, total_padding_r + r, result.shape[r_dim_axis]))
                else:
                    result = dc.op("narrow", [result], (r_dim_axis, 0, total_padding_r + r, result.shape[r_dim_axis]))

                dc.fuse(result)
                return

            elif mode_idx == 2:
                # Reflect mode
                result = activations
 
                if channel_last:
                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
 
                    orig_shape = result.shape
                    result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2]*orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_reflect_sparse_picker(c, r, top, bottom, left, right)
                    spm = dc.tensor(spm)
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    result = dc.op("reshape", [result], (1, orig_shape[-3], orig_shape[-1]+total_padding_r, orig_shape[-2]+total_padding_c))

                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
                else:
                    # import pdb; pdb.set_trace()
                    orig_shape = result.shape
                    if len(orig_shape) == 2:
                        result = dc.op("reshape", [result], (1, orig_shape[-2]*orig_shape[-1]))
                    else:
                        result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2]*orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result]) 
                    spm = create_pad_reflect_sparse_picker(r, c, left, right, top, bottom)
                    spm = dc.tensor(spm) 
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result]) 
                    if len(orig_shape) == 2:
                        result = dc.op("reshape", [result], (orig_shape[-2]+total_padding_r, orig_shape[-1]+total_padding_c))
                    else:
                        result = dc.op("reshape", [result], (1, orig_shape[-3], orig_shape[-2]+total_padding_r, orig_shape[-1]+total_padding_c))

                dc.fuse(result)
                return

    if type == "broadcast":
        if attr[1] == 1:
            dc.fuse(dc.op(Nop.create(), [inputs[0]]))

    if type == "transpose":
        # canonicalize dims to use negative indexing
        dim0, dim1, orig_size = attr
        if dim0 >= 0 or dim1 >= 0:
            if dim0 >= 0:
                dim0 -= inputs[0].shape.len()
            if dim1 >= 0:
                dim1 -= inputs[0].shape.len()
            dc.fuse(dc.op(TransposeTM.create(dim0, dim1, orig_size)), inputs)
            
    if type == "pixel_shuffle":
        result = inputs[0] # (1, C*r*r, H, W)
        orig_shape = result.shape
        if attr[0] != 2:
            raise NotImplementedError("Pixel shuffle decomposition only supports r=2")

        r = attr[0]
        C = orig_shape[-3] // (r * r)
        H = orig_shape[-2]
        W = orig_shape[-1]
        
        result = dc.op("vstack", [result], (r*r,))
        sub_slices = []
        for subsection in range(r):
            sub_slice = dc.op("select", [result], (-2, subsection*r*H, r*H, result.shape[-2]))
            sub_sub_slices = []
            for subsubsection in range(r):
                sub_sub_slices.append(dc.op("select", [sub_slice], (-2, subsubsection*H, H, sub_slice.shape[-2])))
                
            
            curr_sub_sub_slice = sub_sub_slices[0]
            for sub_sub_slice in sub_sub_slices[1:]:
                curr_sub_sub_slice = dc.op("binary_stack", [curr_sub_sub_slice, sub_sub_slice],  (-1,))
                
            sub_slices.append(curr_sub_sub_slice)
        
        curr_sub_slice = dc.op(TransposeTM.create(-2, -1), [sub_slices[0]])
        for sub_slice in sub_slices[1:]:
            sub_slice = dc.op(TransposeTM.create(-2, -1), [sub_slice])
            curr_sub_slice = dc.op("binary_stack", [curr_sub_slice, sub_slice], (-1,))
        
        result = dc.op(TransposeTM.create(-2, -1), [curr_sub_slice])
        dc.fuse(result)
        
    if type == "reshape":
        assert len(inputs) == 1
        input_shape = inputs[0].shape.as_list()
        shape = list(attr)

        if shape == input_shape:
            #dc.fuse(dc.op("nop", [inputs[0]]))
            return

        rank = 0
        while len(shape) < len(input_shape):
            shape.insert(0, 1)
            rank -= 1
        while len(shape) > len(input_shape) and shape[0] == 1:
            shape = shape[1:]
            rank += 1

        is_rank_only_reshape = (shape == input_shape)
        if is_rank_only_reshape and rank != 0:
            result = inputs[0]
            while rank < 0:
                result = dc.op("squeeze", [result], (0,))
                rank += 1
            while rank > 0:
                result = dc.op("unsqueeze", [result], (0, len(result.shape.as_list())))
                rank -= 1
            dc.fuse(result)
            return

def picker_matmul(use_sparse_mm, dc, s, result):
    if use_sparse_mm:                     
        lhs = dc.tensor(s)
        result = dc.op("sparse_matmul", [lhs, result])
    else:
        lhs = dc.tensor(s.to_dense())
        result = dc.op("matmul", [lhs, result])

    return result

def pad_to_tile_dim(n):
    if n % TILE_DIM == 0:
        return n
    return n + TILE_DIM - (n % TILE_DIM)


def decompose_select(attr, dc, inputs):
        orig_shape = inputs[0].shape
        dim, index, length, stride = attr
        if dim >= 0:
            dim -= len(orig_shape)

        result = inputs[0]
        if orig_shape[dim] == length:
            result = dc.op(Nop.create(), [result])
            dc.fuse(result)

        # select on z dim is supported via splice
        elif dim == -3:
            return
            
        # At least one of index, length, or stride is not tile dim aligned, and we are operating on either the x or y dim
        # For example selecting rows 30-35 from tensor of shape (1, 1, 64, 128)
        elif not (index % TILE_DIM == length % TILE_DIM == stride % TILE_DIM == 0) and dim in [-2, -1] and stride == orig_shape[dim]:
            assert len(attr) == 4, "Select should have 4 attributes"
            x = result
            x = dc.op("pad_tile", [x], (-2 , orig_shape[-2]))
            x = dc.op("pad_tile", [x], (-1 , orig_shape[-1]))
            
            cols = []
            size = len(range(index, orig_shape[dim], stride))*len(range(index, index + length))
            for offset in range(0, orig_shape[dim], stride):
                for i in range(index, index + length):
                    if offset + i < orig_shape[dim] or stride == orig_shape[dim]:
                        cols.append(offset + i)
            
            rows = list(range(len(cols)))
            vals = [1.0]*len(cols)
            
            spm = torch.sparse_coo_tensor((rows, cols), vals, (align_up_tile(size), x.shape[dim]))
            if len(result.shape) > 2 and result.shape[-3] > 1:
                spm = torch.stack([spm]*result.shape[-3], -3).unsqueeze(0)
            spm = dc.tensor(spm)
            
            is_x_select = dim == -1
            if is_x_select:
                    x = dc.op(TransposeTM.create(-2, -1), [x])
                   
            result = dc.op("sparse_matmul", [spm, x])
             
            if is_x_select:
                result = dc.op(TransposeTM.create(-2, -1), [result])
                
            if is_x_select:
                result = dc.op("narrow", [result], (-1, 0, size, result.shape[-1]))
                result = dc.op("narrow", [result], (-2, 0, orig_shape[-2], result.shape[-2]))
            else:
                
                result = dc.op("narrow", [result], (-1, 0, orig_shape[-1], result.shape[-1]))
                result = dc.op("narrow", [result], (-2, 0, size, result.shape[-2]))

            dc.fuse(result)
                        
            return

def decompose_xy_flatten_reshape(inputs, dc, orig_shape, attr):
    use_sparse_mm = True
    result = inputs[0]
    result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
    result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))

    if orig_shape[-3] > 1:
        result = dc.op("hstack", [result], (orig_shape[-3],))

    padded_shape = result.shape
    r_new = padded_shape[-1] * orig_shape[-2] // (padded_shape[-1] // TILE_DIM)
    pad_for_factrization = False
    sparse_r_padding = ast.literal_eval(os.environ.get('PYBUDA_PAD_SPARSE_MM', "{}"))
    if orig_shape[-2] % TILE_DIM and orig_shape[-1] % TILE_DIM and orig_shape[-2] in sparse_r_padding:
        pad_for_factrization = True
        padded_r_new = sparse_r_padding[orig_shape[-2]] * TILE_DIM
        cols = torch.arange(r_new//TILE_DIM)
        rows = cols * TILE_DIM
        fl_spm =  torch.sparse_coo_tensor([rows.tolist(), cols.tolist()], torch.ones(cols.shape[0]), (padded_r_new, result.shape[-2]), dtype=torch.float32,) 
    else:
        fl_spm = create_reshape_flatten_sparse_picker_matrix(result.shape[-2], r_new)
    result = picker_matmul(use_sparse_mm, dc, fl_spm, result)
    use_sparse_mm = True

    if orig_shape[-3] > 1:
        result = dc.op("hslice", [result], (orig_shape[-3],))
        # result = dc.op(Buffer.create(), [result]) # HW workaround for: tenstorrent/budabackend#656
        
    rt = align_up_tile(r_new) // TILE_DIM
    if pad_for_factrization:
        rt = sparse_r_padding[orig_shape[-2]]
    result = dc.op("vslice", [result], (rt,))
    # result = dc.op(Buffer.create(), [result]) # HW workaround for: tenstorrent/budabackend#656
    result = dc.op("hstack", [result], (rt,))

    if orig_shape[-3] > 1:
        # result = dc.op(Buffer.create(), [result]) # HW workaround for: tenstorrent/budabackend#656
        result = dc.op("vstack", [result], (orig_shape[-3],))


    if orig_shape[-1] % TILE_DIM:
        result = dc.op(TransposeTM.create(-2, -1), [result])
        if pad_for_factrization:
            num_pads = orig_shape[-2]
            cols = []
            [cols.extend((torch.arange(0, orig_shape[-1]) + (pad_to_tile_dim(orig_shape[-1]) * pad)).tolist()) for pad in range(num_pads)]
            cols = torch.tensor(cols)
            rows = torch.arange(num_pads * orig_shape[-1])
            num_rows = num_pads * orig_shape[-1]
            s = torch.sparse_coo_tensor([rows.tolist(), cols.tolist()], torch.ones(cols.shape[0]), (num_rows, result.shape[-2]), dtype=torch.float32,)
        else:
            s = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, orig_shape[-1], pad_to_tile_dim(orig_shape[-1]))
        result = picker_matmul(use_sparse_mm, dc, s, result)
        result = dc.op(TransposeTM.create(-2, -1), [result])

    while len(result.shape) > len(attr):
        result = dc.op("squeeze", [result], (0,))

    while len(result.shape) < len(attr):
        result = dc.op("unsqueeze", [result], (0, len(result.shape.as_list())))

    if orig_shape[-3] > 1:
        s = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, 1, TILE_DIM)
        result = picker_matmul(use_sparse_mm, dc, s, result)
    else:
        result = dc.op("narrow", [result], (-2, 0, 1, result.shape[-2]))
    
    return result
        
def decompose_xy_unflatten(inputs, dc, orig_shape, attr):
    result = inputs[0]
    use_sparse_mm = True
    # Pick out Z dim values from Y and expand the X dimension result matrix by TILE_DIM times
    # Original matrix:
    #               |   0   |   1   |  ...  | LY=len(Y) - 1
    #               -------------------------------------
    #       0       | A0,0  | A0,1  |  ...  |   A0,LY   |
    #       1       | A1,0  | A1,1  |  ...  |   A1,LY   |
    #      ...      |  ...  |  ...  |  ...  |    ...    |
    #  LX=len(X)-1  | ALX,0 | ALX,1 |  ...  |   ALX,LY  |
    #
    # Picker matrix is in the following format:
    #                       |   0   |   1   |   2   |  ...  |   LX  | 
    #                       ----------------------------------------|
    #        0              |   1   |   0   |   0   |  ...  |   0   |
    #        1              |   0   |   0   |   0   |   0   |   0   |
    #       ...             |   0   |   0   |   0   |   0   |   0   |
    #     TILE_DIM          |   0   |   1   |   0   |   0   |   0   |
    #       ...             |   0   |   0   |   0   |   0   |   0   |
    #    2*TILE_DIM         |   0   |   0   |   1   |   0   |   0   |  
    #       ...             |   0   |   0   |   0   |   0   |   0   |  
    #    LX*TILE_DIM        |   0   |   0   |   0   |   0   |   1   |
    # LX*TILE_DIM+TILE_DIM-1|   0   |   0   |   0   |   0   |   0   |

    s_pick_z = create_reshape_flatten_sparse_picker_matrix(orig_shape[-2], orig_shape[-2] * TILE_DIM)  
    result = picker_matmul(use_sparse_mm, dc, s_pick_z, result)

    # Result matrix is in the following format:
    #                       |   0   |   1   |  ...  |   LY  |
    #                       ---------------------------------
    #           0           | A0,0  | A0,1  |  ...  | A0,LY |
    #           ...         |   0   |   0   |   0   |   0   |
    #       TILE_DIM        | A1,0  | A1,1  |  ...  | A1,LY |
    #           ...         |   0   |   0   |   0   |   0   |
    #       2*TILE_DIM      | A2,0  | A2,1  |  ...  | A2,LY |
    #           ...         |   0   |   0   |   0   |   0   |
    #       LX*TILE_DIM     | ALX,0 | ALX,1 |   0   | ALX,LY|
    #           ...         |   0   |   0   |   0   |   0   |
    # LX*TILE_DIM+TILE_DIM-1|   0   |   0   |   0   |   0   |

    _orig_shape = result.shape
    # Pad X dimension to TILE_DIM size
    if _orig_shape[-2] % TILE_DIM != 0:
        result = dc.op("pad_tile", [result], (-2, _orig_shape[-2]))
    
    # Pad Y dimension to TILE_DIM size
    if _orig_shape[-1] % TILE_DIM != 0:
        result = dc.op("pad_tile", [result], (-1, _orig_shape[-1]))
    
    # Transpose the result matrix
    result = dc.op(TransposeTM.create(-2, -1), [result])
    slice_factor = _orig_shape[-1] // attr[-1]

    # After matrix transpose, the result is in the following format:
    #       |   0   |  ...  |  TILE_DIM |  ...  |   2*TILE_DIM  |  ...  |  LX*TILE_DIM  |  ...  |LX*TILE_DIM+TILE_DIM-1 |
    #       --------------------------------------------------------------------------------------------------------------
    #   0   | A0,0  |   0   |   A1,0    |  ...  |   A2,0        |  ...  |   ALX,0       |  ...  |           0           |
    #   1   | A0,1  |   0   |   A1,1    |  ...  |   A2,1        |  ...  |   ALX,1       |  ...  |           0           |
    #  ...  |  ...  |   0   |   ...     |  ...  |   ...         |  ...  |   ...         |  ...  |           0           |
    #  LY   | A0,LY |   0   |   A1,LY   |  ...  |   A2,LY       |  ...  |   ALX,LY      |  ...  |           0           |

    # If new X\Y dimensions aren't divisible by TILE_DIM, we need to padd the resulting matrix
    if attr[-1] % TILE_DIM != 0 or attr[-2] % TILE_DIM != 0:
        padded_dim = (math.ceil(attr[-1] / TILE_DIM) * TILE_DIM)
        num_tiles = attr[-2] if attr[-1] < TILE_DIM else (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
        new_size = num_tiles * padded_dim

        cols = torch.arange(orig_shape[-1]).tolist()
        rows = []
        for i in range(attr[-2]):
            rows.extend((torch.arange(attr[-1]) + (i * padded_dim)).tolist())

        # For example, picker matrix is in the following format, where LNY represents new Y dimension:
        #                   |   0   |   1   |  ...  |   LNY | LNY+1 | LNY+2 | ... |
        #                   -------------------------------------------------------
        #      0            |   1   |   0   |   0   |  ...  |   0   |   0   | ... |    
        #      1            |   0   |   1   |   0   |  ...  |   0   |   0   | ... |
        #     ...           |   0   |   0   |   1   |  ...  |   0   |   0   | ... |
        #     LNY           |   0   |   0   |   0   |   1   |   0   |   0   | ... |
        #     ...           |   0   |   0   |   0   |   0   |   0   |   0   | ... |
        #   padded_dim      |   0   |   0   |   0   |   0   |   1   |   0   | ... |
        #  padded_dim+1     |   0   |   0   |   0   |   0   |   0   |   1   | ... |
        #     ...           |   0   |   0   |   0   |   0   |   0   |   0   | ... |
        s_pad_with_zero = torch.sparse_coo_tensor(
            [rows, cols],
            torch.ones(len(cols)),
            (new_size, result.shape[-2]),
            dtype=torch.float32,
        )
        result = picker_matmul(use_sparse_mm, dc, s_pad_with_zero, result)

    # Slice out Z dim
    result = dc.op(TransposeTM.create(-2, -1), [result])
    if orig_shape[-2] > 1:
        result = dc.op("vslice", [result], (orig_shape[-2], ))
    elif len(result.shape) == 2:
        result = dc.op("unsqueeze", [result], (0, 2,))
    _orig_shape = result.shape
    slice_factor = attr[-2] if attr[-1] < TILE_DIM else (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
    result = dc.op(TransposeTM.create(-2, -1), [result])

    # Slice out row size
    result = dc.op("vslice", [result], (slice_factor, ))
    result = dc.op(TransposeTM.create(-2, -1), [result])
    result = dc.op("vstack", [result], (slice_factor * _orig_shape[-3], ))

    # Pick out mulitple rows and pack them into tiles
    s = create_reshape_flatten_sparse_picker_matrix(slice_factor * attr[-3], result.shape[-2]).transpose(-1, -2)
    result = picker_matmul(use_sparse_mm, dc, s, result)

    if (_orig_shape[-3] > 1) and (attr[-3] > 1):
        result = dc.op("vslice", [result], (attr[-3],))

    if attr[-1] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
    if attr[-2] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))   
    return result

def decompose_non_tile_dim_aligned_vslice(inputs, dc, orig_shape, attr):
    result = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
    use_sparse_mm = True

    slice_factor = attr[-3]
    result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
    result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
    if attr[-2] % TILE_DIM != 0 or orig_shape[-2] % TILE_DIM != 0:
        padded_dim = (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
        num_tiles = attr[-3] if attr[-2] < TILE_DIM else (math.ceil(attr[-3] / TILE_DIM) * TILE_DIM)
        new_size = num_tiles * padded_dim

        cols = torch.arange(orig_shape[-2]).tolist()
        rows = []
        for i in range(attr[-3]):
            rows.extend((torch.arange(attr[-2]) + (i * padded_dim)).tolist())

        spm = torch.sparse_coo_tensor(
            [rows, cols],
            torch.ones(len(cols)),
            (new_size, result.shape[-2]),
            dtype=torch.float32,
        )
        if attr[-2] >= TILE_DIM:
            spm1 = create_flattened_padding_removal_sparse_picker_matrix(spm.shape[-2], 0, slice_factor*padded_dim, spm.shape[-2])
            spm = torch.sparse.mm(spm1, spm)

        result = picker_matmul(use_sparse_mm, dc, spm, result)
    
    result = dc.op("vslice", [result], (slice_factor, ))
    if attr[-1] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
    if attr[-2] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))
        
    return result

def decompose_post_optimize(type, attr, dc, inputs):
    if type == "reshape":
        
        orig_shape = []
        unbroadcast_shape = []
        for i in range(inputs[0].shape.len()):
            orig_shape.append(inputs[0].shape[i])
        for i in range(inputs[0].unbroadcast_shape.len()):
            unbroadcast_shape.append(inputs[0].unbroadcast_shape[i])
        orig_attr = attr.copy()

        while len(orig_shape) < 4:
            orig_shape.insert(0, 1)
        while len(attr) < 4:
            attr.insert(0, 1)

        if orig_shape == attr:
            return
        
        result = inputs[0]

        if all([dim == 1 for dim in unbroadcast_shape]):
            x_broadcast = attr[-1]
            y_broadcast = attr[-2]

            add_vslice = False
            if len(attr) > 2 and attr[-3] != 1:
                y_broadcast *= attr[-3]
                add_vslice = True

            if x_broadcast != 1:
                result = dc.op("broadcast", [result], [-1, x_broadcast, True], False)

            if y_broadcast != 1:
                result = dc.op("broadcast", [result], [-2, y_broadcast, True], False)
            
            if add_vslice:
                result = dc.op("vslice", [result], (attr[-3],))
        
        #                                      if z = 1 its a shrink y grow x reshape
        elif len(orig_shape) > 1 and orig_shape[-2] > 1 and orig_shape[-3] > 1 and all([dim == 1 for dim in attr[:-1]]):
            use_sparse_mm = True
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            padded_shape = result.shape

            if len(orig_shape) > 2 and orig_shape[-3] != 1:
                result = dc.op("vstack", [result], (orig_shape[-3], ))
                if orig_shape[-2] % TILE_DIM:
                    s = create_padding_shift_sparse_picker_matrix(orig_shape[-3]*orig_shape[-2], orig_shape[-3], result.shape[-2])#.transpose(-2, -1)
                    result = picker_matmul(use_sparse_mm, dc, s, result)

            padded_shape = result.shape
            r_new = TILE_DIM * padded_shape[-2]
            
            s = create_reshape_flatten_sparse_picker_matrix(padded_shape[-2], r_new)
            result = picker_matmul(use_sparse_mm, dc, s, result)

            rt = align_up_tile(r_new) // TILE_DIM
            result = dc.op("vslice", [result], (rt,))
            result = dc.op(Buffer.create(), [result]) # HW workaround for: tenstorrent/budabackend#656
            result = dc.op("hstack", [result], (rt,))
        
            result = dc.op(TransposeTM.create(-2, -1), [result])

            s = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, orig_shape[-1], padded_shape[-1])
            rows = torch.arange(0, align_up_tile(attr[-1])).tolist()
            cols = rows
            s2 = torch.sparse_coo_tensor((rows, cols), torch.ones(len(rows)), (len(rows), s.shape[-2]), dtype=torch.float32)
            s = torch.sparse.mm(s2, s)
            result = picker_matmul(use_sparse_mm, dc, s, result)

            result = dc.op(TransposeTM.create(-2, -1), [result])
            result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))
            result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))

        elif len(orig_shape) > 2 and len(attr) > 2 and len(orig_shape) == len(attr) and orig_shape[-1] == 1 \
            and not(attr[-2] % TILE_DIM == 0 and attr[-1] % TILE_DIM == 0) and orig_shape[-3] == attr[-3] and orig_shape[-4] == 1 \
            and not(orig_shape[-1] == attr[-2] and attr[-1] == orig_shape[-2]):

            slice_factor = orig_shape[-2] // attr[-1]
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            
            cols = torch.arange(orig_shape[-2]).tolist()
            rows = torch.arange(0, len(cols)*TILE_DIM, TILE_DIM).tolist()

            spm = torch.sparse_coo_tensor((rows, cols), torch.ones(len(cols)), (len(cols)*TILE_DIM, result.shape[-2]), dtype=torch.float32)
            result = picker_matmul(True, dc, spm, result)
            
            result = dc.op("vslice", [result], (orig_shape[-2],))
            if result.shape[-3] > 1:
                result = dc.op("hstack", [result], (attr[-1],))
            if result.shape[-3] > 1:
                result = dc.op("vstack", [result], (attr[-2],))
            
            spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, 1, TILE_DIM, True)
            result = picker_matmul(True, dc, spm, result)
            
            result = dc.op(TransposeTM.create(-2, -1), [result])
            spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, 1, TILE_DIM, True)
            result = picker_matmul(True, dc, spm, result)
            result = dc.op(TransposeTM.create(-2, -1), [result])
            
            result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))
            result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))

        elif len(orig_shape) > 2 and len(attr) > 2 and len(orig_shape) == len(attr) and (orig_shape[-1] == 1 or orig_shape[-2] == 1) \
            and attr[-2] % TILE_DIM == 0 and attr[-1] % TILE_DIM == 0 and orig_shape[-3] == attr[-3] and orig_shape[-4] == 1:
            # (1, Z, 1, X) -> (1, Z, y_, x_) where y_ and x_ are divisible by tile dim
            # (1, Z, Y, 1) -> (1, Z, y_, x_) where y_ and x_ are divisible by tile dim
            if orig_shape[2] == 1:
                result = dc.op(TransposeTM.create(-2, -1), [result])
                slice_factor = orig_shape[-1] // attr[-1]
                result = dc.op("pad_tile", [result], (-2, orig_shape[-1]))
                result = dc.op("pad_tile", [result], (-1, orig_shape[-2]))
            else:
                slice_factor = orig_shape[-2] // attr[-1]
                result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
                result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))

            use_sparse_mm = True


            result = dc.op("vslice", [result], (slice_factor, ))
            result = dc.op(TransposeTM.create(-2, -1), [result])
            result = dc.op("vstack", [result], (slice_factor * orig_shape[-3], ))
            s = create_reshape_flatten_sparse_picker_matrix(attr[-2] * attr[-3], result.shape[-2]).transpose(-1, -2)
            result = picker_matmul(use_sparse_mm, dc, s, result)

            if (orig_shape[-3] > 1) and (attr[-3] > 1):
                result = dc.op("vslice", [result], (attr[-3],))
    
        elif (len(orig_shape) > 2 and len(attr) > 2 and len(orig_shape) == len(attr) and orig_shape[-2] == attr[-3] and orig_shape[-3] == 1 and orig_shape[-1] == attr[-1] * attr[-2] and attr[-1] != orig_shape[-3] \
            and attr[-2] != 1 and attr[-1] != 1 and len(unbroadcast_shape) == 2 and attr[-2] < attr[-1] and orig_shape[-2] % TILE_DIM == 0 and orig_shape[-1] % TILE_DIM == 0):
            # [A, B] --> [1, A', B', C'] where:
            #    A == A',  B == B' * C',  B' != 1, C' != 1, A % TILE_DIM == 0, B % TILE_DIM == 0
            use_sparse_mm = True            

            slice_factor = attr[-2]  
            result = dc.op("hslice", [result,], (slice_factor,))
            result = dc.op(TransposeTM.create(-3, -2, 1,), [result,])

        elif (len(orig_shape) > 2 and len(attr) > 2 and len(orig_shape) == len(attr) and orig_shape[-2] == attr[-3] and orig_shape[-3] == 1 and orig_shape[-1] == attr[-1] * attr[-2] and attr[-1] != orig_shape[-3] \
            and attr[-2] != 1 and attr[-1] != 1): # and orig_shape[-2] % TILE_DIM == 0):
            # [1, 1, 32, 1024] -> [1, 32, 32, 32]
            #     [1, 32, 1, 1024] -> [1, 32, 1024, 1] -> [1, 1024, 32, 1] -> [1, 1024, 1, 32] -> [1, 1, 1024, 32] -> picker -> vslice
            # [1, 1, 1280, 4] -> [1, 1280, 2, 2]
            result = decompose_xy_unflatten(inputs, dc, orig_shape, attr)

        elif len(orig_shape) > 2 and attr[-1] != 1 and orig_shape[-1] != 1 and orig_shape[-2] != 1 and attr[-2] == orig_shape[-3] and attr[-1] == orig_shape[-1] * orig_shape[-2]:
            # xy flatten
            # [1, 32, 64, 64] -> [1, 1, 32, 4096]
            # [1, 1, 32, 32] -> [1, 1, 1, 1024]
            result = decompose_xy_flatten_reshape(inputs, dc, orig_shape, attr)

        elif len(orig_shape) > 2 and orig_shape[-3] == attr[-1] and orig_shape[-2] == attr[-2] == 1 and (orig_shape[-1] == 1 or orig_shape[-3] == 1):
            # XZ transpose
            result = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            use_sparse_mm = True

            if orig_shape[-1] == 1:
                # [1, 1280, 1, 1,] -> [1, 1, 1, 1280]
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op("pad_tile", [result], (-2, orig_shape[-1]))

                # Vstack to y dim
                result = dc.op("vstack", [result], (orig_shape[-3], ))

                s = create_reshape_flatten_sparse_picker_matrix(orig_shape[-3], result.shape[-2]).transpose(-1, -2)
                result = picker_matmul(use_sparse_mm, dc, s, result)

                result = dc.op(TransposeTM.create(-2, -1), [result])

            elif orig_shape[-3] == 1:
                # [1, 1, 1, 1280,] -> [1, 1280, 1, 1]
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op("pad_tile", [result], (-2, orig_shape[-1]))

                cols = torch.arange(orig_shape[-1])
                rows = cols * TILE_DIM
                # Resulted row dim is TILE_DIM * (orig_shape[-1]), orig_shape[-1] doesnt need to be tile aligned
                s = torch.sparse_coo_tensor(
                    [rows.tolist(), cols.tolist()],
                    torch.ones(cols.shape[0]),
                    (orig_shape[-1] * TILE_DIM, result.shape[-2]),
                    dtype=torch.float32,
                )
                result = picker_matmul(use_sparse_mm, dc, s, result)

                result = dc.op("vslice", [result], (attr[-3],))
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op("narrow", [result], (-1, 0, 1, 1))

        elif len(orig_shape) > 2 and orig_shape[-3] == attr[-2] and orig_shape[-1] == attr[-1] == 1 and (orig_shape[-2] == 1 or orig_shape[-3] == 1):
            # YZ transpose
            result = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            use_sparse_mm = True

            if orig_shape[-2] == 1:
                # [1, 1280, 1, 1,] -> [1, 1, 1280, 1]
                result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))

                # Vstack to y dim
                result = dc.op("vstack", [result], (orig_shape[-3], ))

                s = create_reshape_flatten_sparse_picker_matrix(orig_shape[-3], result.shape[-2]).transpose(-1, -2)
                result = picker_matmul(use_sparse_mm, dc, s, result)

            elif orig_shape[-3] == 1:
                # [1, 1, 1280, 1,] -> [1, 1280, 1, 1]
                result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))

                cols = torch.arange(orig_shape[-2])
                rows = cols * TILE_DIM
                # Resulted row dim is TILE_DIM * (orig_shape[-2]), orig_shape[-2] doesnt need to be tile aligned
                s = torch.sparse_coo_tensor(
                    [rows.tolist(), cols.tolist()],
                    torch.ones(cols.shape[0]),
                    (orig_shape[-2] * TILE_DIM, result.shape[-2]),
                    dtype=torch.float32,
                )
                result = picker_matmul(use_sparse_mm, dc, s, result)

                result = dc.op("vslice", [result], (attr[-3],))
                result = dc.op("narrow", [result], (-2, 0, 1, 1))

        elif orig_shape[0] == attr[0] and orig_shape[1] == attr[1] and orig_shape[2] == attr[3] and orig_shape[3] == attr[2] and (attr[2] == 1 or attr[3] == 1):
            result = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            result = dc.op(TransposeTM.create(-2, -1), [result])

        elif orig_shape[-4] == attr[-4] and orig_shape[-3] == 1 and orig_shape[-2] == attr[-3] * attr[-2] and orig_shape[-1] == attr[-1] and orig_shape[-3] != attr[-3]:
            # VSLICE non-tile_dim aligned
            result = decompose_non_tile_dim_aligned_vslice(inputs, dc, orig_shape, attr)

        elif orig_shape[-4] == attr[-4] and attr[-3] == 1 and attr[-2] == orig_shape[-3] * orig_shape[-2] and orig_shape[-1] == attr[-1] and orig_shape[-3] != attr[-3]:
            # VSTACK non-tile_dim aligned
            result = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            use_sparse_mm = True
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))

            padded_shape = result.shape
            slice_factor = orig_shape[-3]

            result = dc.op("vstack", [result], (slice_factor, ))

            if orig_shape[-2] % TILE_DIM != 0:
                # Pick out multiple rows in a tile
                num_rows = (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
                num_tiles = orig_shape[-3]
                rows = torch.arange(attr[-2]).tolist()
                cols = []

                for i in range(num_tiles):
                    cols.extend((torch.arange(orig_shape[-2]) + (i * padded_shape[-2])).tolist())

                s_pick_multi_row = torch.sparse_coo_tensor(
                    [rows, cols],
                    torch.ones(len(cols)),
                    (num_rows, result.shape[-2]),
                    dtype=torch.float32,
                )

                result = picker_matmul(use_sparse_mm, dc, s_pick_multi_row, result)
    
            if attr[-1] % TILE_DIM != 0:
                result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
            if attr[-2] % TILE_DIM != 0:
                result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))

        elif len(orig_shape) > 2 and orig_shape[-1] % attr[-1] == 0 and ((orig_shape[-3] == attr[-3] and orig_shape[-2] != attr[-2] and orig_shape[-2] != 1 and orig_shape[-2]*orig_shape[-1] == attr[-2]*attr[-1]) or \
            (orig_shape[-3] != attr[-3] and orig_shape[-2] == 1 and attr[-3] == 1 and orig_shape[-3] != attr[-2] and orig_shape[-3] != 1 and orig_shape[-3]*orig_shape[-1] == attr[-2]*attr[-1])) \
            and attr[-2] != 1  and orig_shape[-1] > attr[-1]:
            # (1, z, 5, 12) -> (1, z, 20, 3)
            use_sparse_mm = False
        
            if orig_shape[-3] != attr[-3]:
                result = dc.op(TransposeTM.create(-3, -2, 1), [result])
                orig_shape[-3], orig_shape[-2] = orig_shape[-2], orig_shape[-3]

            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            padded_shape = result.shape
            slice_factor = orig_shape[-1] // attr[-1]

            spm = create_reshape_flatten_sparse_picker_matrix(padded_shape[-2], TILE_DIM*padded_shape[-2]).unsqueeze(0).unsqueeze(0)
            result = picker_matmul(use_sparse_mm, dc, spm, result)
            result = dc.op("vslice", [result], (result.shape[-2] // TILE_DIM, ))

            if orig_shape[-1] % TILE_DIM == 0 and attr[-1] % TILE_DIM == 0:
                result = dc.op("hslice", [result], (slice_factor, ))
                result = dc.op("vstack", [result], (result.shape[-3] // orig_shape[-3],))
            else:
                slice_size = orig_shape[-1] // slice_factor
                padded_slice_size = align_up_tile(slice_size)
                rows = []
                [rows.extend(torch.arange(s*padded_slice_size, s*padded_slice_size + slice_size).tolist()) for s in range(slice_factor)]
                cols = torch.arange(orig_shape[-1]).tolist()
                spm = torch.sparse_coo_tensor([rows, cols], torch.ones(len(cols)), (slice_factor*padded_slice_size, padded_shape[-1]), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = picker_matmul(use_sparse_mm, dc, spm, result)
                result = dc.op("vslice", [result], (slice_factor, ))
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op("vstack", [result], (result.shape[-3] // orig_shape[-3],))
            
            spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, 1, TILE_DIM)
            spm = torch.nn.functional.pad(spm.to_dense(), (0, 0, 0, align_up_tile(spm.shape[-2]) - spm.shape[-2]), mode='constant', value=0).to_sparse().unsqueeze(0).unsqueeze(0)

            result = picker_matmul(use_sparse_mm, dc, spm, result)

            if not align_up_tile(attr[-2]) == align_up_tile(result.shape[-2]):
                idxs = torch.arange(attr[-2]).tolist()
                spm = torch.sparse_coo_tensor([idxs, idxs], torch.ones(attr[-2]), (attr[-2], result.shape[-2]), dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
                result = picker_matmul(use_sparse_mm, dc, spm, result)

            if attr[-1] % TILE_DIM != 0:
                result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
            if attr[-2] % TILE_DIM != 0:
                result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))

        elif len(orig_shape) > 2 and orig_shape[-2] % attr[-2] == 0 and ((orig_shape[-3] == attr[-3] and orig_shape[-2] != attr[-2] and orig_shape[-2] != 1 and orig_shape[-1]*orig_shape[-2] == attr[-1]*attr[-2]) or \
            (orig_shape[-3] != attr[-3] and orig_shape[-2] == 1 and attr[-3] == 1 and orig_shape[-3] != attr[-2] and orig_shape[-3] != 1 and orig_shape[-1]*orig_shape[-3] == attr[-1]*attr[-2])) \
            and attr[-2] != 1 and orig_shape[-1] < attr[-1]:
            # (1, z, 20, 3) -> (1, z, 5, 12)
            use_sparse_mm = False
        
            if orig_shape[-3] != attr[-3]:
                result = dc.op(TransposeTM.create(-3, -2, 1), [result])
                orig_shape[-3], orig_shape[-2] = orig_shape[-2], orig_shape[-3]

            slice_factor = orig_shape[-2] // attr[-2]

            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            padded_shape = result.shape

            spm = create_reshape_flatten_sparse_picker_matrix(padded_shape[-2], TILE_DIM*padded_shape[-2]).unsqueeze(0).unsqueeze(0)
            result = picker_matmul(use_sparse_mm, dc, spm, result)
            result = dc.op("vslice", [result], (padded_shape[-2],))

            if orig_shape[-2] % TILE_DIM == 0:
                result = dc.op("hstack", [result], (slice_factor,))
                result = dc.op("vstack", [result], (result.shape[-3] // orig_shape[-3],))

            else:
                result = dc.op("vstack", [result], (result.shape[-3] // padded_shape[-3],))
                new_len = result.shape[-2] - TILE_DIM*(result.shape[-2] // TILE_DIM - orig_shape[-2])
                spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0,  new_len, result.shape[-2]).unsqueeze(0).unsqueeze(0)
                result = picker_matmul(use_sparse_mm, dc, spm, result)
                result = dc.op("vslice", [result], (orig_shape[-2],))
                result = dc.op("hstack", [result], (slice_factor,))
                result = dc.op("vstack", [result], (orig_shape[-2] // slice_factor,))

            if orig_shape[-1] % TILE_DIM != 0:
                spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-1], 0, orig_shape[-1], padded_shape[-1])
                spm = torch.nn.functional.pad(spm.to_dense(), (0, 0, 0, align_up_tile(spm.shape[-2]) - spm.shape[-2]), mode='constant', value=0).to_sparse().unsqueeze(0).unsqueeze(0)
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = picker_matmul(use_sparse_mm, dc, spm, result)
                result = dc.op(TransposeTM.create(-2, -1), [result])
                
            spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, 1, TILE_DIM).unsqueeze(0).unsqueeze(0)
            spm = torch.nn.functional.pad(spm.to_dense(), (0, 0, 0, align_up_tile(spm.shape[-2]) - spm.shape[-2]), mode='constant', value=0).to_sparse()

            result = picker_matmul(use_sparse_mm, dc, spm, result)

            if attr[-1] % TILE_DIM != 0:
                result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
            if attr[-2] % TILE_DIM != 0:
                result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))
            
        elif len(orig_shape) > 2 and len(attr) > 2 and orig_shape[-3] == 1 and attr[-3] == orig_shape[-2] and attr[-2] == orig_shape[-1]:
            slice_factor = result.shape[-2]
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            
            cols = torch.arange(orig_shape[-2]).tolist()
            rows = torch.arange(0, len(cols)*TILE_DIM, TILE_DIM).tolist()
            spm = torch.sparse_coo_tensor((rows, cols), torch.ones(len(cols)), (len(cols)*TILE_DIM, result.shape[-2]), dtype=torch.float32)
            result = picker_matmul(True, dc, spm, result)
            
            result = dc.op("vslice", [result], (slice_factor,))
            result = dc.op(TransposeTM.create(-2, -1), [result])
            
            result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
            result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))
        
            
        elif len(orig_shape) == 4 and len(orig_attr) < 4 and orig_shape[-4] == 1 and orig_shape[-3] * orig_shape[-2] == attr[-3] * attr[-2] and orig_shape[-1] == attr[-1]:
            # example: (1, 1024, 4, 128) -> (1, 128, 32, 128)
            # TODO: this is a temporary solution for decomposing reshape in pytorch implementation of Grouped Query Attention
            # in this particular case, padding is not needed, while adding padding would increase the input tensor size 8 times (dim -2 going from 4 to 32)
            # in the future, hard constraint of divisibility of R dim by TILE_DIM should be removed, but removing it currently causes some models to fail compilation
            # hint: look into function convert_reshape_into_vslice_or_vstack_if_possible
            result = dc.op('vstack', [result], (orig_shape[-3] // attr[-3],))

        elif len(orig_shape) == 4 and len(attr) == 4 and orig_shape[-4] == attr[-4] and orig_shape[-1] == attr[-1] and attr[-3] == orig_shape[-2] * orig_shape[-3] and attr[-2] == 1:
            # example: (1, 6, 8, 128) -> (1, 48, 1, 128)
            # TODO: this is a temporary solution for decomposing reshape in pytorch implementation of Grouped Query Attention
            # in this particular case, padding is not needed, while adding padding would increase the resulting tensor size 32 times (dim -2 going from 1 to 32)
            # in the future, hard constraint of divisibility of R dim by TILE_DIM should be removed, but removing it currently causes some models to fail compilation
            # hint: look into function convert_reshape_into_vslice_or_vstack_if_possible
            result = dc.op('vslice', [result], (orig_shape[-2],))
            
        elif (len(orig_shape) < 4 or (len(orig_shape) == 4 and orig_shape[0] == 1)) \
              and (len(attr) < 4 or (len(attr) == 4 and attr[0] == 1)): # General reshape (only support for w == 1)
            # Ex (1, 38, 38, 84) -> (1, 2, 2888, 21)
            
            # xy flatten: (1, 1, 38, 3192)
            result = decompose_xy_flatten_reshape(inputs, dc, orig_shape, (*attr[:-3], 1, attr[-3], orig_shape[-2]*orig_shape[-1]))
            if orig_shape[-3] > 1:
                # xy flatten: (1, 1, 1, 121296)
                result = decompose_xy_flatten_reshape([result], dc, result.shape, (*attr[:-3], 1, 1, orig_shape[-3]*orig_shape[-2]*orig_shape[-1]))
            
            # xy unflatten: (1, 1, 5776, 21)
            if attr[-2] > 1:
                result = decompose_xy_unflatten([result], dc, result.shape, (attr[:-3], 1, attr[-3]*attr[-2], attr[-1]))
                if attr[-3] > 1:
                    # vslice: (1, 2, 2888, 21)
                    result = decompose_non_tile_dim_aligned_vslice([result], dc, result.shape, attr)
            elif attr[-3] > 1: # Make it just an hslice instead
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = decompose_non_tile_dim_aligned_vslice([result], dc, result.shape, (*attr[:-2], attr[-1], attr[-2]))
                result = dc.op(TransposeTM.create(-2, -1), [result])
        else:
            return

        while len(result.shape) > len(orig_attr):
            result = dc.op("squeeze", [result], (0,))

        while len(result.shape) < len(orig_attr):
            result = dc.op("unsqueeze", [result], (0, len(result.shape.as_list())))
    
        dc.fuse(result)
        return

    elif type == "transpose":
        orig_shape = inputs[0].shape
        dim1, dim2, z_dim_slice = attr
        if len(orig_shape) > 2 and dim1 == -3 and dim2 == -1 and ((len(orig_shape) == 4 and orig_shape[-4] == 1) or len(orig_shape) < 4):
        # XZ transpose
            result = inputs[0]
            use_sparse_mm = True
            
            result = inputs[0]
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
            result = dc.op(TransposeTM.create(-2, -1), [result])

            if result.shape[-3] > 1:
                result = dc.op("vstack", [result], (orig_shape[-3], ))
            i_spm = create_sparse_interleave_picker_matrix(result.shape[-2], orig_shape[-1], orig_shape[-3])
            result = picker_matmul(use_sparse_mm, dc, i_spm, result)
            
            result = dc.op("vslice", [result], (orig_shape[-1], ))
            result = dc.op(TransposeTM.create(-2, -1), [result])

            result = dc.op("narrow", [result], (-2, 0, orig_shape[-2], result.shape[-2]))
            result = dc.op("narrow", [result], (-1, 0, orig_shape[-3], result.shape[-1]))
            
            dc.fuse(result)

        elif dim1 == -3 and dim2 == -2 and ((len(orig_shape) == 4 and orig_shape[0] == 1) or len(orig_shape) == 3):
            # YZ transpose
            result = inputs[0]
            use_sparse_mm = True
            
            result = inputs[0]
            result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
            result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))

            if result.shape[-3] > 1:
                result = dc.op("vstack", [result], (orig_shape[-3], ))
            i_spm = create_sparse_interleave_picker_matrix(result.shape[-2], orig_shape[-2], orig_shape[-3])
            result = picker_matmul(use_sparse_mm, dc, i_spm, result)
            
            if orig_shape[-2] > 1:
                result = dc.op("vslice", [result], (orig_shape[-2], ))

            result = dc.op("narrow", [result], (-2, 0, orig_shape[-3], result.shape[-2]))
            result = dc.op("narrow", [result], (-1, 0, orig_shape[-1], result.shape[-1]))
            
            dc.fuse(result)

    # TODO: remove once backend support is available
    elif type == "select":
        decompose_select(attr, dc, inputs)

    elif type == "repeat":
        sizes = attr
        result = inputs[0]
        for dim, factor in enumerate(sizes):
            neg_idx = dim - len(inputs[0].shape) # Use negative indexing
            if factor == 1:
                continue
            result = dc.op("repeat_dim", [result], (neg_idx, factor))
        dc.fuse(result)
    
    elif type == "repeat_dim":
        axis = attr[0]
        if inputs[0].shape[axis] % TILE_DIM != 0 and (axis == -2 or axis == -1):
            # Decompose repeat to spase mm
            orig_shape = inputs[0].shape.as_list()
            orig_dim = orig_shape[axis]
            target_dim_size = inputs[0].shape[axis] * attr[1]
            rounded_target_dim = align_up_tile(target_dim_size)
            if axis == -2:
                result = inputs[0]
                use_sparse_mm = True
                result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
                i_spm = create_repeat_sparse_picker_matrix(orig_dim, attr[1])
                result = picker_matmul(use_sparse_mm, dc, i_spm, result)
                result = dc.op("narrow", [result], (-2, 0, target_dim_size, result.shape[-2]))
            elif axis == -1:
                result = inputs[0]
                use_sparse_mm = True
                result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
                result = dc.op(TransposeTM.create(-2, -1), [result])
                i_spm = create_repeat_sparse_picker_matrix(orig_dim, attr[1])
                result = picker_matmul(use_sparse_mm, dc, i_spm, result)
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op("narrow", [result], (-1, 0, target_dim_size, result.shape[-1]))
            else:
                assert False
    
            dc.fuse(result)
    
    elif type == "hslice":
        input_shape = inputs[0].shape.as_list() 
        post_dim = input_shape[-1] // attr[0]
        result = inputs[0]
        if post_dim % TILE_DIM != 0:
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op("pad_tile", [result,], (-2, input_shape[-2]))
            cols = []
            pad_post_dim = align_up_tile(post_dim)
            pad_input_dim = pad_post_dim * attr[0]
            for i in range(attr[0]):
                cols.extend(torch.arange(i*pad_post_dim, i*pad_post_dim+post_dim).tolist())
            spm = torch.sparse_coo_tensor(
                [cols, torch.arange(input_shape[-1]).tolist()],
                torch.ones(input_shape[-1]),
                (pad_input_dim, input_shape[-1]),
                dtype=torch.float32,
            )

            while len(result.shape) < 3:
                result = dc.op("unsqueeze", [result,], (0, len(result.shape.as_list())))

            spm = torch.stack([spm]*result.shape[-3], -3).unsqueeze(0)
            result = dc.op(TransposeTM.create(-2, -1), [result,])
            result = picker_matmul(True, dc, spm, result)
            result = dc.op(TransposeTM.create(-2, -1), [result,])
            result = dc.op("hslice", [result,], attr)
            result = dc.op("narrow", [result,], (-1, 0, post_dim, result.shape[-1]))
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op("narrow", [result,], (-2, 0, input_shape[-2], result.shape[-2]))
            dc.fuse(result)
        elif input_shape[-2] % TILE_DIM != 0:
            result = dc.op("pad_tile", [result,], (-2, input_shape[-2]))
            result = dc.op("hslice", [result,], attr)
            result = dc.op("narrow", [result,], (-2, 0, input_shape[-2], result.shape[-2]))
            dc.fuse(result)

    elif type == "hstack":
        input_shape = inputs[0].shape.as_list()
        result = inputs[0]
        if input_shape[-1] % TILE_DIM != 0:
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op("pad_tile", [result,], (-2, input_shape[-2]))
            output_dim = input_shape[-1] * attr[0]
            pad_output_dim = align_up_tile(input_shape[-1]) * attr[0]
            result = dc.op("pad_tile", [result,], (-1, input_shape[-1]))
            result = dc.op("hstack", [result,], attr)
            rows = []
            pad_input_dim = align_up_tile(input_shape[-1])
            for i in range(attr[0]):
                rows.extend(torch.arange(i*pad_input_dim, i*pad_input_dim+input_shape[-1]).tolist())
            spm = torch.sparse_coo_tensor(
                [torch.arange(output_dim).tolist(), rows],
                torch.ones(output_dim),
                (output_dim, pad_output_dim),
                dtype=torch.float32,
            )
            spm = torch.stack([spm]*result.shape[-3], -3).unsqueeze(0)
            result = dc.op(TransposeTM.create(-2, -1), [result,])
            result = picker_matmul(True, dc, spm, result)
            result = dc.op(TransposeTM.create(-2, -1), [result,])
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op("narrow", [result,], (-2, 0, input_shape[-2], result.shape[-2]))
            dc.fuse(result)
        elif input_shape[-2] % TILE_DIM != 0:
            result = dc.op("pad_tile", [result,], (-2, input_shape[-2]))
            result = dc.op("hstack", [result,], attr)
            result = dc.op("narrow", [result,], (-2, 0, input_shape[-2], result.shape[-2]))
            dc.fuse(result)

    elif type == "vslice":
        input_shape = inputs[0].shape.as_list()
        post_dim = input_shape[-2] // attr[0]
        result = inputs[0]
        if post_dim % TILE_DIM != 0:
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op("pad_tile", [result,], (-1, input_shape[-1]))
            cols = []
            pad_post_dim = align_up_tile(post_dim)
            pad_input_dim = pad_post_dim * attr[0]
            for i in range(attr[0]):
                cols.extend(torch.arange(i*pad_post_dim, i*pad_post_dim+post_dim).tolist())
            spm = torch.sparse_coo_tensor(
                [cols, torch.arange(input_shape[-2]).tolist()],
                torch.ones(input_shape[-2]),
                (pad_input_dim, input_shape[-2]),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)
            spm = torch.cat([spm]*result.shape[-3], -3)
            result = picker_matmul(True, dc, spm, result)
            result = dc.op("vslice", [result,], attr)
            result = dc.op("narrow", [result,], (-2, 0, post_dim, result.shape[-2]))
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op("narrow", [result,], (-1, 0, input_shape[-1], result.shape[-1]))
            dc.fuse(result)
        elif input_shape[-1] % TILE_DIM != 0:
            result = dc.op("pad_tile", [result,], (-1, input_shape[-1]))
            result = dc.op("vslice", [result,], attr)
            result = dc.op("narrow", [result,], (-1, 0, input_shape[-1], result.shape[-1]))
            dc.fuse(result)

    elif type == "vstack":
        input_shape = inputs[0].shape.as_list()
        result = inputs[0]
        if input_shape[-2] % TILE_DIM != 0:
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op("pad_tile", [result,], (-1, input_shape[-1]))
            output_dim = input_shape[-2] * attr[0]
            pad_output_dim = align_up_tile(input_shape[-2]) * attr[0]
            result = dc.op("pad_tile", [result,], (-2, input_shape[-2]))
            result = dc.op("vstack", [result,], attr)
            rows = []
            pad_input_dim = align_up_tile(input_shape[-2])
            for i in range(attr[0]):
                rows.extend(torch.arange(i*pad_input_dim, i*pad_input_dim+input_shape[-2]).tolist())
            spm = torch.sparse_coo_tensor(
                [torch.arange(output_dim).tolist(), rows],
                torch.ones(output_dim),
                (output_dim, pad_output_dim),
                dtype=torch.float32,
            ).coalesce().unsqueeze(0).unsqueeze(0)
            spm = torch.cat([spm]*result.shape[-3], -3)
            result = picker_matmul(True, dc, spm, result)
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op("narrow", [result,], (-1, 0, input_shape[-1], result.shape[-1]))
            dc.fuse(result)
        elif input_shape[-1] % TILE_DIM != 0:
            result = dc.op("pad_tile", [result,], (-1, input_shape[-1]))
            result = dc.op("vstack", [result,], attr)
            result = dc.op("narrow", [result,], (-1, 0, input_shape[-1], result.shape[-1]))
            dc.fuse(result)

    return


def decompose_post_autograd(type, attr, dc, inputs):
    if type == "select":
        decompose_select(attr, dc, inputs)      
        return

    if type == "reshape":
        assert len(inputs) == 1
        input_shape = inputs[0].shape.as_list()
        shape = list(attr)

        if shape == input_shape:
            #dc.fuse(dc.op(Nop.create(), [inputs[0]]))
            return

        rank = 0
        while len(shape) < len(input_shape):
            shape.insert(0, 1)
            rank -= 1
        while len(shape) > len(input_shape) and shape[0] == 1:
            shape = shape[1:]
            rank += 1

        is_rank_only_reshape = (shape == input_shape)
        if is_rank_only_reshape and rank != 0:
            result = inputs[0]
            while rank < 0:
                result = dc.op("squeeze", [result], (0,))
                rank += 1
            while rank > 0:
                result = dc.op("unsqueeze", [result], (0, len(result.shape.as_list())))
                rank -= 1
            dc.fuse(result)
            return


def decompose_reshape(attr, dc, inputs):
        orig_attr = attr.copy()
        while len(attr) > 4:
            assert attr[0] == 1, "Cannot eliminate non-singleton dimension"
            attr = attr[1:]
        while len(attr) < 4:
            attr.insert(0, 1)

        # Pad shape to 4D before lowering
        orig_shape = []
        for i in range(inputs[0].shape.len()):
            orig_shape.append(inputs[0].shape[i])
        while len(orig_shape) < 4:
            orig_shape.insert(0, 1)

        if orig_shape == attr:
            return

        assert len(attr) == 4, "Reshape should have 4 attributes"
        # 
        # Decompose reshapes that are effectively xy transposes of a vector as such
        if orig_shape[0] == attr[0] and orig_shape[1] == attr[1] and orig_shape[2] == attr[3] and orig_shape[3] == attr[2] and (attr[2] == 1 or attr[3] == 1):
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            xy_transpose = dc.op(TransposeTM.create(-2, -1), [inp])
            output = squeeze_output_for_reshape_decomp(dc, xy_transpose, orig_attr)
            dc.fuse(output)
            return

        if (orig_shape[0] == attr[0] and orig_shape[2] == attr[1] and orig_shape[1] == 1 and orig_shape[3] == attr[2] * attr[3] and attr[-1] != orig_shape[1]
            and orig_shape[-1] % TILE_DIM == 0 and (orig_shape[-1] // attr[2]) % TILE_DIM == 0): # HSLICE needs to remember actual size ; resulted shape must be divisible by TILE_DIM
            # [1, 1, 16, 2048] -> [1, 16, 16, 128] hslice + yz transpose
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            slice_amount = attr[2]
            if slice_amount != 1:
                hslice = dc.op("hslice", [inp], (slice_amount,))
            else:
                hslice = inp
            yz_transpose = dc.op(TransposeTM.create(-3, -2, attr[1]), [hslice,])
            output = squeeze_output_for_reshape_decomp(dc, yz_transpose, orig_attr)
            dc.fuse(output)
            return

        if (orig_shape[0] == attr[0] and attr[1] == 1 and attr[3] == orig_shape[2] * orig_shape[3] and orig_shape[-1] != attr[1] and orig_shape[1] == attr[2] and orig_shape[2] != attr[1]
           and orig_shape[-1] % TILE_DIM == 0): # Hstack needs to remember actual size
            #  [1, 16, 16, 128] -> [1, 1, 16, 2048] yz transpose + hstack
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[-2]), [inp])
            stack_amount = orig_shape[2]
            hstack = dc.op("hstack", (yz_transpose,), (stack_amount,))
            output = squeeze_output_for_reshape_decomp(dc, hstack, orig_attr)
            dc.fuse(output)
            return

        if (orig_shape[0] == attr[0] and orig_shape[1] == attr[1] and orig_shape[2] * orig_shape[3] == attr[2] and attr[3] == 1 and orig_shape[2] != attr[3]
            and orig_shape[-1] % TILE_DIM == 0): # Vstack needs to remember actual size
            # [1, 768, 12, 64] -> [1, 768, 768, 1]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[-2]), [inp,])
            vstack = dc.op("hstack", (yz_transpose,), (orig_shape[-2],))
            xz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[1]), (vstack,))
            output = squeeze_output_for_reshape_decomp(dc, xz_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[0] and attr[1] == orig_shape[3] and attr[2] == orig_shape[1] and attr[3] == orig_shape[2] and attr[1] == 1:
            # [1, 768, 768, 1] -> [1, 1, 768, 768]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            xz_transpose = dc.op(TransposeTM.create(-3, -1, orig_shape[3]), [inp,])
            xy_transpose = dc.op(TransposeTM.create(-2, -1), (xz_transpose,))
            output = squeeze_output_for_reshape_decomp(dc, xy_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[0] and orig_shape[2] == attr[1] and orig_shape[3] == attr[2] and orig_shape[1] == attr[3] and orig_shape[1] == 1:
            # [1, 1, 512, 101] -> [1, 512, 101, 1]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[2]), [inp,])
            xy_transpose = dc.op(TransposeTM.create(-2, -1), (yz_transpose,))
            output = squeeze_output_for_reshape_decomp(dc, xy_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[0] and orig_shape[2] == attr[1] and orig_shape[1] == attr[2] and orig_shape[3] == attr[3] and (orig_shape[1] == 1 or orig_shape[2] == 1):
            # [1, 1, 512, 101] -> [1, 512, 1, 101]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[2]), [inp,])
            output = squeeze_output_for_reshape_decomp(dc, yz_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[0] and orig_shape[1] == 1 and orig_shape[2] == attr[1] * attr[2] and orig_shape[3] == attr[3] and orig_shape[2] % TILE_DIM == 0 and (orig_shape[2] // attr[1]) % TILE_DIM == 0:
            # [1, 1, 6144, 768] -> [1, 48, 128, 768] Y dim must be divisible by tile_dim
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            vslice = dc.op("vslice", [inp,], (attr[1],))
            output = squeeze_output_for_reshape_decomp(dc, vslice, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[0] and attr[1] == 1 and attr[2] == orig_shape[1] * orig_shape[2] and orig_shape[3] == attr[3] and orig_shape[2] % TILE_DIM == 0:
            # [1, 48, 128, 768] -> [1, 1, 6144, 768]  Y dim must be divisible by tile_dim
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            vstack = dc.op("vstack", [inp,], (orig_shape[1],))
            output = squeeze_output_for_reshape_decomp(dc, vstack, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == 1 and orig_shape[1] == attr[0] and orig_shape[2] == attr[1] and orig_shape[3] == attr[2] * attr[3] and orig_shape[3] % TILE_DIM == 0 and attr[2] != 1 and attr[3] != 1 and attr[3] % TILE_DIM == 0:
            # [1, 48, 128, 768] -> [48, 128, 12, 64]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            wz_transpose = dc.op(TransposeTM.create(0, 1), [inp,])
            if attr[2] != 1:
                hslice = dc.op("hslice", [wz_transpose,], (attr[2],))
            else:
                hslice = wz_transpose
            yz_transpose = dc.op(TransposeTM.create(-3, -2, attr[1],), [hslice,])
            output = squeeze_output_for_reshape_decomp(dc, yz_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] * orig_shape[1] == attr[2] and orig_shape[2] * orig_shape[3] == attr[3] and attr[0] == 1 and attr[1] == 1 and orig_shape[0] != 1 and orig_shape[2] != 1:
            # [48, 128, 12, 64] -> [1, 1, 6144, 768]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[2]), [inp,])
            hstack = dc.op("hstack", [yz_transpose,], (orig_shape[2],))
            wz_transpose = dc.op(TransposeTM.create(0, 1, -1,), [hstack,])
            vstack = dc.op("vstack", [wz_transpose,], (orig_shape[0],))
            output = squeeze_output_for_reshape_decomp(dc, vstack, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[0] and orig_shape[1] == attr[3] and orig_shape[2] == attr[2] and orig_shape[3] == attr[1] and (orig_shape[1] == 1 or orig_shape[3] == 1):
            # [1, 1, 1, 96] -> [1, 96, 1, 1]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            xz_transpose = dc.op(TransposeTM.create(-3, -1, orig_shape[3]), [inp,])
            output = squeeze_output_for_reshape_decomp(dc, xz_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[1] and orig_shape[1] == attr[2] and orig_shape[2] == attr[0] and orig_shape[3] == attr[3] and orig_shape[2] == 1 and orig_shape[3] == 1:
            # [176, 192, 1, 1] -> [1, 176, 192, 1]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yz_transpose = dc.op(TransposeTM.create(-3, -2, orig_shape[2]), [inp,])
            wz_transpose = dc.op(TransposeTM.create(0, 1), [yz_transpose,])
            output = squeeze_output_for_reshape_decomp(dc, wz_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[1] and orig_shape[1] == attr[0] and orig_shape[2] == attr[2] and orig_shape[3] == attr[3]:
            # [1, 32, 1, 384] -> [32, 1, 1, 384]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            wz_transpose = dc.op(TransposeTM.create(-4, -3), [inp,])
            output = squeeze_output_for_reshape_decomp(dc, wz_transpose, orig_attr)
            dc.fuse(output)
            return

        if orig_shape[0] == attr[2] and orig_shape[1] == attr[1] and orig_shape[2] == attr[0] and orig_shape[3] == attr[3] and attr[1] == 1: # if attr[1] != 1, this isnt a transpose
            # [1, 1, 32, 128] -> [32, 1, 1, 128]
            inp = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
            yw_transpose = dc.op(TransposeTM.create(-4, -2, orig_shape[2]), [inp,])
            output = squeeze_output_for_reshape_decomp(dc, yw_transpose, orig_attr)
            dc.fuse(output)
            return
