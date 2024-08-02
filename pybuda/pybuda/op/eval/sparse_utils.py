# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import math
import numpy as np
import os
import torch
import torch.nn.functional;
from loguru import logger
import pybuda
from pybuda.utils import align_up_tile, align_up, round_up_div, clamp
from ...pybudaglobal import TILE_DIM
from ...tensor import narrow_buda_tensor_to_pytorch, pad_pytorch_tensor_to_buda
from pybuda._C import DataFormat, compress_sparse_tensor_and_strip_info, SparseCOO, SparseBUDA, MathFidelity
from math import gcd


def conv2d_padding_to_canonical(padding, kernel_size):
    # current implementation is without dilation

    assert isinstance(kernel_size,int) or isinstance(kernel_size,tuple) , "Unsupported kernel size"
    kH, kW = kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size)

    if isinstance(padding, int):
        return [padding] * 4
    elif isinstance(padding, str):
        assert padding == "same" , "Unsupported padding"
        padding = [kW // 2] * 2 + [kH // 2] * 2
        if kW % 2 == 0:
            padding[1] -= 1
        if kH % 2 == 0:
            padding[3] -= 1
        return padding
    elif isinstance(padding, tuple) or isinstance(padding, list):
        if len(padding) == 2:
            return [padding[1]] * 2 + [padding[0]] * 2
        elif len(padding) == 4:
            return list(padding)
        else:
            raise AssertionError ("Unsupported padding")
    else:
        raise AssertionError ("Unsupported padding")


def conv3d_padding_to_canonical(padding, kernel_size):
    # current implementation is without dilation

    assert isinstance(kernel_size,int) or isinstance(kernel_size,tuple) , "Unsupported kernel size"
    kD, kH, kW = kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size,kernel_size)

    if isinstance(padding, int):
        return [padding] * 6
    elif isinstance(padding, str):
        assert padding == "same" , "Unsupported padding"
        padding = [kW // 2] * 2  + [kH // 2] * 2 + [kD // 2] * 2
        if kW % 2 == 0:
            padding[1] -= 1
        if kH % 2 == 0:
            padding[3] -= 1
        if kD % 2 == 0:
            padding[5] -= 1
        return padding
    elif isinstance(padding, tuple) or isinstance(padding, list):
        if len(padding) == 2:
            return [padding[1]] * 3 + [padding[0]] * 3
        elif len(padding) == 6:
            return list(padding)
        else:
            raise AssertionError ("Unsupported padding")
    else:
        raise AssertionError ("Unsupported padding")



def calculate_conv2d_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1, ceil_mode=False
):
    if isinstance(stride, int):
        stride = [stride] * 2

    assert len(padding) == 4 and all(isinstance(x, int) for x in padding), "Padding should be list of four ints"
    
    if dilation != 1:
        logger.warning("Dilation values other than 1 are not supported for kernel fracturing.")

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Padding is [left, right, top, bottom]
    if ceil_mode:
        y = (
            math.ceil(
                (original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) / stride[0]
            )
            + 1
        )
        x = (
            math.ceil(
                (original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1]
            )
            + 1
        )
    else:
        y = (original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
        x = (original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return y, x


def calculate_conv3d_output_dimensions(
    original_z, original_y, original_x, kernel_size, stride, padding, dilation=1, ceil_mode=False
):
    if isinstance(stride, int):
        stride = [stride] * 3

    assert len(padding) == 6 and all(isinstance(x, int) for x in padding), "Padding should be list of six ints"

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Padding is [left, right, top, bottom]
    if ceil_mode:
        z = (
            math.ceil(
                (original_z + padding[4] + padding[5] - dilation * (kernel_size[0] - 1) - 1) / stride[0]
            )
            + 1
        )
        y = (
            math.ceil(
                (original_y + padding[2] + padding[3] - dilation * (kernel_size[1] - 1) - 1) / stride[1]
            )
            + 1
        )
        x = (
            math.ceil(
                (original_x + padding[0] + padding[1] - dilation * (kernel_size[2] - 1) - 1) / stride[2]
            )
            + 1
        )
    else:
        z = (original_z + padding[4] + padding[5] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
        y = (original_y + padding[2] + padding[3] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
        x = (original_x + padding[0] + padding[1] - dilation * (kernel_size[2] - 1) - 1) // stride[2] + 1
    return z, y, x


def calculate_conv2d_transpose_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1, output_padding=0
):
    if isinstance(padding, int):
        padding = [padding] * 4

    y = (
        (original_y - 1) * stride
        - (padding[2] + padding[3])
        + dilation * (kernel_size[0] - 1)
        + 1
        + output_padding
    )
    x = (
        (original_x - 1) * stride
        - (padding[0] + padding[1])
        + dilation * (kernel_size[1] - 1)
        + 1
        + output_padding
    )
    return y, x


def calculate_conv2d_prestride_weights_and_padding(weights, original_y, original_x, stride, padding):
    shape_only = isinstance(weights, list) or isinstance(weights, tuple)

    if isinstance(padding, int):
        padding = [padding] * 4
    if isinstance(stride, int):
        stride = [stride] * 2
    if shape_only:
        weights = torch.zeros(weights, requires_grad=False)

    kernel_size = (weights.shape[-2], weights.shape[-1])

    pre_strided_weights = []
    weights_left_pad = padding[0] % stride[1]
    weights_left_pad = stride[1] if weights_left_pad == 0 else weights_left_pad
    weights_top_pad = padding[2] % stride[0]
    weights_top_pad = stride[0] if weights_top_pad == 0 else weights_top_pad
    for y in range(stride[0]):
        for x in range(stride[1]):
            pre_strided_weights.append(torch.nn.functional.pad(weights, (stride[1] - weights_left_pad, x, stride[0] - weights_top_pad, y))[:, :, y::stride[0], x::stride[1]])

    pre_strided_weights = torch.cat(pre_strided_weights, dim=-3)

    def pad_right_bottom(padding, stride, original_shape, output_shape, kernel_size):
        # Output height (width) for convolution can be calculated as:
        #     y = (original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
        #
        # If we know all the parameters of the prestrided conv (except the right/bottom padding), we can solve for right/bottom padding
        #
        # First we remove dilation as it's always 1:
        #     y = (original_y + padding[2] + padding[3] - (kernel_size[0] - 1) - 1) // stride[0] + 1
        # Simplify:
        #     y = (original_y + padding[2] + padding[3] - kernel_size[0]) // stride[0] + 1
        #     (y - 1) * stride = original_y + padding[2] + padding[3] - kernel_size[0]
        #     (y - 1) * stride - original_y - padding[2] + kernel_size[0] = padding[3]
        return (output_shape - 1) * stride - original_shape - padding + kernel_size

    output_shape = calculate_conv2d_output_dimensions(
        original_y=original_y,
        original_x=original_x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1)

    pad_left = round_up_div(padding[0], stride[1])
    pad_top = round_up_div(padding[2], stride[0])
    pad_right = pad_right_bottom(pad_left, 1, round_up_div(original_x, stride[1]), output_shape[1], pre_strided_weights.shape[-1])
    pad_bottom = pad_right_bottom(pad_top, 1, round_up_div(original_y, stride[0]), output_shape[0], pre_strided_weights.shape[-2])

    pre_strided_padding = (
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
    )

    if shape_only:
        pre_strided_weights = tuple(pre_strided_weights.shape)
    return pre_strided_weights, pre_strided_padding


def _calculate_pad_for_ceil_mode_single_dim(original_dim, out_dim, kernel_size, stride, pad_prefix, pad_suffix, dilation=1):
    assert isinstance(original_dim, int)
    assert isinstance(out_dim, int)
    assert isinstance(kernel_size, int)
    assert isinstance(stride, int)
    assert isinstance(pad_prefix, int)
    assert isinstance(pad_suffix, int)

    # If the dimension we're looking at is width, pad_prefix is padding on the left, pad_suffix is padding on the right
    # For height, they're top and bottom, respectively

    # ceil_mode adds padding only when last sliding kernel window would start within pad_prefix or input area, but not
    # pad_suffix area
    # If the suffix padding is already >= than kernel_size, don't pad for ceil_mode
    if pad_suffix >= kernel_size:
        return 0

    # Equation to calculate output size (in single dimension) for convs and pooling ops:
    #   out_dim = floor((in_dim + total_padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    # 
    # For out_dim, we can plug in the size we would get if we were to do ceil_mode=True, solve for total_padding, plug 
    # in all the other params, and subtract the original padding from the result in order to calculate how much extra
    # padding we need to add so that we get the desired out_dim
    # Additionally, we can remove the floor() (or ceil()) function from the equation to get the exact padding
    #
    # Solve for total_padding:
    #   total_padding = stride * (out_dim - 1) - in_dim + dilation * (kernel_size - 1) + 1
    # 
    # Calculate amount of padding to add:
    #   padding_to_add = total_padding - pad_prefix - pad_suffix
    total_padding = stride * (out_dim - 1) - original_dim + dilation * (kernel_size - 1) + 1
    padding_to_add = total_padding - pad_prefix - pad_suffix
    assert padding_to_add >= 0, "Uh-oh, ceil_mode math doesn't check out :/"

    return padding_to_add


def calculate_pad_for_ceil_mode(original_y, original_x, kernel_size, stride, padding, dilation=1):
    if isinstance(stride, int):
        stride = [stride] * 2

    assert len(padding) == 4 and all(isinstance(x, int) for x in padding), "Padding should be list of four ints"

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    y_out, x_out = calculate_conv2d_output_dimensions(original_y, original_x, kernel_size, stride, padding, dilation, ceil_mode=True)

    pad_right = _calculate_pad_for_ceil_mode_single_dim(
        original_dim=original_x,
        out_dim=x_out,
        kernel_size=kernel_size[1],
        stride=stride[1],
        pad_prefix=padding[0],
        pad_suffix=padding[1],
        dilation=1
    )

    pad_bottom = _calculate_pad_for_ceil_mode_single_dim(
        original_dim=original_y,
        out_dim=y_out,
        kernel_size=kernel_size[0],
        stride=stride[0],
        pad_prefix=padding[2],
        pad_suffix=padding[3],
        dilation=1
    )

    return pad_right, pad_bottom


def create_conv2d_picker_matrix(y, x, y_shift, x_shift, stride, tile_align=False):
    total_shift = (y_shift * x) + x_shift
    dident = torch.torch.nn.functional.pad(torch.ones(x), (x_shift, -x_shift))
    dident = dident.repeat(y)
    dident = torch.diag(dident)
    dident = torch.torch.nn.functional.pad(dident, (-total_shift, total_shift))
    dident = dident.view(-1, x, y * x)[::stride, ::stride]
    dident = dident.reshape(-1, y * x)
    if tile_align:
        dident = torch.nn.functional.pad(
            dident,
            (
                0,
                align_up_tile(dident.shape[1]) - dident.shape[1],
                0,
                align_up_tile(dident.shape[0]) - dident.shape[0],
            ),
        )
    return dident


def create_conv2d_sparse_picker_matrix(
    y, x, y_shift, x_shift, k_y, k_x, stride, padding, dilation, tile_align=False, pad_x_only=False, sparse_r_pad=0, sparse_c_pad=0, is_convtranspose2d=False, yout_transpose=None, xout_transpose=None
):
    cols = torch.arange(start=1, end=y * x + 1).view(y, x)

    # pad
    cols = torch.nn.functional.pad(cols, padding)
    # shift
    shift_y = dilation * ((k_y - 1) // 2 - y_shift)
    shift_x = dilation * ((k_x - 1) // 2 - x_shift)
    cols = torch.nn.functional.pad(cols, (-shift_x, shift_x, -shift_y, shift_y))
    # stride
    cols = cols[::stride[0], ::stride[1]]
    # clamp to output dims
    out_y, out_x = calculate_conv2d_output_dimensions(
        y, x, [k_y, k_x], stride, padding, dilation
    )
    if is_convtranspose2d:
        out_y = yout_transpose
        out_x = xout_transpose

    cols = torch.nn.functional.pad(
        cols, (0, out_x - cols.shape[1], 0, out_y - cols.shape[0])
    )

    cols = cols.reshape(-1)
    rows = torch.arange(cols.shape[0])
    rows = rows.index_select(0, cols.nonzero().flatten())
    cols = cols.index_select(0, cols.nonzero().flatten())
    cols -= 1

    if pad_x_only:
        # Channel last conv
        sparse_r = align_up_tile(out_x) * out_y
        sparse_c = align_up_tile(x) * y
    elif tile_align:
        sparse_r = align_up_tile(out_y * out_x)
        sparse_c = align_up_tile(y * x)
        if sparse_r_pad:
            sparse_r_tile = align_up_tile(out_y * out_x) // 32
            sparse_r = (sparse_r_tile + sparse_r_pad) * 32
        if sparse_c_pad:
            sparse_c += sparse_c_pad * 32
    else:
        sparse_r = out_y * out_x
        sparse_c = y * x
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (sparse_r, sparse_c),
        dtype=torch.float32,
    ).coalesce()


def create_dilate2d_sparse_picker_matrix(y, x, dilation, tile_align=False):
    rows = torch.arange(y * x).view(y, x)
    rows *= dilation
    for i in range(y):
        rows[i] += i * x * dilation
    rows = rows.view(y * x)
    cols = torch.arange(y * x)
    sparse_r = (y * dilation) * (x * dilation)
    sparse_c = y * x
    if tile_align:
        sparse_r = align_up_tile(sparse_r)
        sparse_c = align_up_tile(sparse_c)
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (sparse_r, sparse_c),
        dtype=torch.float32,
    )


def create_avg_pool2d_count_include_pad_False_picker_matrix(y, x, k_y, k_x, stride, padding, tile_align=False):
    """
    When avg_pool2d has its parameter `count_include_pad` set to False, we will treat it as True and then try to fix it
    afterwards.
    This function creates a picker matrix (used by sparse mm) that is subsequently used to correct the output of the
    avg_pool2d. It does so by finding output pixels that took padding pixels into account, undoing the division by
    multiplying (by the same factor) and then divinding by the correct factor (count non padding pixels).

    TODO: Writing this function as a conv was natural, but it can probably be written to use less memory.
    """

    # Create an input tensor of 0s, and add padding of 1s
    padding_mask = torch.zeros([y, x], dtype=torch.uint8).view(1, 1, y, x)
    padding_mask = torch.nn.functional.pad(padding_mask, padding, value=1)

    # Perform a convolution in order to get how many padding pixels figure into each output pixel
    weights = torch.ones(1, 1, k_y, k_x, dtype=torch.uint8)
    picker = torch.nn.functional.conv2d(padding_mask, weights, stride=stride)[0, 0]

    # Figure out what to multiply each output pixel with, in order to get correct avg_pool2d
    onehot = picker.bool().byte()  # int(bool(picker))
    kernel_volume = k_y * k_x
    picker = onehot * kernel_volume / (kernel_volume - picker)
    picker = picker + 1 - onehot  # this adds 1s where 0s are

    # Transform into sparse picker
    picker = picker.reshape(-1)
    rows_cols = torch.arange(picker.shape[0]).tolist()

    if tile_align:
        picker_dim = align_up_tile(picker.shape[0])
    else:
        picker_dim = picker.shape[0]

    return torch.sparse_coo_tensor(
        indices=[rows_cols, rows_cols],
        values=picker,
        size=(picker_dim, picker_dim),
        dtype=torch.float32,
    ).coalesce()


def create_index_sparse_picker_matrix(r, start, stop, stride, tile_align=False):
    length = stop - start
    rows = torch.arange(r).narrow(0, start, length) - start
    cols = torch.arange(r).narrow(0, start, length)
    rows = torch.div(rows[::stride], stride, rounding_mode="floor")
    cols = cols[::stride]
    sparse_r = min(r, round_up_div(length, stride))
    sparse_c = r
    if tile_align:
        sparse_r = align_up_tile(sparse_r)
        sparse_c = align_up_tile(sparse_c)
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (sparse_r, sparse_c),
        dtype=torch.float32,
    )


def create_reshape_flatten_sparse_picker_matrix(orig_r, new_r, tile_dim=TILE_DIM):
    cols = torch.arange(new_r//tile_dim)
    rows = cols * tile_dim
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (new_r, orig_r),
        dtype=torch.float32,
    )


def create_reshape_flatten_sparse_picker_matrix_narrower(orig_r, new_r, org_length, tile_dim=TILE_DIM):
    cols = torch.arange(orig_r)
    rows = torch.tensor([])
    for i in range(0, new_r, tile_dim): 
        rows = torch.cat([rows, (torch.arange(org_length) + i)])  
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (new_r, orig_r),
        dtype=torch.float32,
    )

def create_flattened_padding_removal_sparse_picker_matrix(r, start, stop, length, align_up_rows=False, align_up_cols=False):
    num_pads = r // length
    cols = []
    [cols.extend((torch.arange(start, stop) + (length * pad)).tolist()) for pad in range(num_pads)]
    cols = torch.tensor(cols)
    rows = torch.arange(num_pads * stop)
    num_rows = num_pads * stop if not align_up_rows else align_up_tile(num_pads * stop)
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (num_rows, r if not align_up_cols else align_up_tile(r)),
        dtype=torch.float32,
    )

def create_padding_shift_sparse_picker_matrix(length, slices, padded_length):
    rows = torch.arange(0, length).tolist()
    cols = []
    for i in range(slices):
        lo = i*align_up_tile(length//slices)
        hi = lo + length//slices
        col = torch.arange(lo, hi)
        cols = cols + col.tolist()

    return torch.sparse_coo_tensor(
        [rows, cols],
        torch.ones(len(cols)),
        (padded_length, align_up_tile(cols[-1]+1)),
        dtype=torch.float32
    )


def create_real_row_sparse_picker_matrix(orig_x, padded_y):
    cols = torch.arange(orig_x)
    rows = cols * TILE_DIM
    # Resulted row dim is TILE_DIM * (orig_shape[-1]), orig_shape[-1] doesnt need to be tile aligned
    spm = torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (orig_x * TILE_DIM, padded_y),
        dtype=torch.float32,
    )
    return spm

def create_repeat_sparse_picker_matrix(orig_x, repeat):
    cols = torch.arange(orig_x).tolist() * repeat
    rows = torch.arange(orig_x * repeat).tolist()

    spm = torch.sparse_coo_tensor(
        [rows, cols],
        torch.ones(len(rows)),
        (align_up_tile(orig_x * repeat), align_up_tile(orig_x)),
        dtype=torch.float32,
    )
    return spm


def create_sparse_interleave_picker_matrix(length, orig_x, orig_z):
    
    def create_rows(orig_x, orig_z):
        rows = []
        curr_row = 0
        for i in range(orig_x):
            rows.append(torch.arange(curr_row, curr_row+orig_z).tolist())
            curr_row += align_up_tile(orig_z)
        return rows
    
    def create_cols(length, orig_x):
        cols = []
        for i in range(orig_x):
            curr_cols = []
            for j in range(length//align_up_tile(orig_x)):
                vals = [i+ j*align_up_tile(orig_x)]
                curr_cols += vals

            cols.append(curr_cols)
        return cols

    rows = create_rows(orig_x, orig_z)
    cols = create_cols(length, orig_x)
    matrix = torch.sparse_coo_tensor([rows[0], cols[0]],torch.ones(len(cols[0])), (align_up_tile(rows[-1][-1]+1), length), dtype=torch.float32)
    for i in range(1, orig_x):
        matrix += torch.sparse_coo_tensor([rows[i], cols[i]],torch.ones(len(cols[i])), (align_up_tile(rows[-1][-1]+1), length), dtype=torch.float32)

    return matrix

def transpose_sparse_picker_matrix(sparse):
    sparse_r = sparse.shape[-2]
    sparse_c = sparse.shape[-1]
    has_w = len(sparse.shape) == 4
    has_z = len(sparse.shape) >= 3
    if not has_z:
        sparse = sparse.unsqueeze(0)
    if has_w:
        assert sparse.shape[0] == 1
        sparse = sparse.select(0, 0)
    transposed = []
    for z in range(sparse.shape[0]):
        s = sparse.select(0, z)
        s = s.coalesce()
        rows, cols = s.indices()
        transposed.append(
            torch.sparse_coo_tensor(
                [cols.tolist(), rows.tolist()],
                s.values(),
                (sparse_c, sparse_r),
                dtype=s.dtype,
            )
        )
    transposed = torch.stack(transposed)
    if not has_z:
        assert transposed.shape[0] == 1
        transposed = transposed.select(0, 0)
    if has_w:
        transposed = transposed.unsqueeze(0)
    return transposed


def bcast_sparse_picker_matrix(sparse, dim, size):
    assert len(sparse.shape) - dim == 3, "Only support sparse bcast on Z"
    assert sparse.shape[dim] == 1
    has_w = len(sparse.shape) == 4
    while len(sparse.shape) > 2:
        sparse = sparse.select(0, 0)
    sparse = torch.stack([sparse] * size)
    if has_w:
        sparse = sparse.unsqueeze(0)
    return sparse


def get_sparse_picker_matrix_max_span(sparse: torch.Tensor):
    while len(sparse.shape) < 4:
        sparse = sparse.unsqueeze(0)
    _, zdim, y, x = sparse.shape

    max_span = -1
    sparse = sparse.select(dim=0, index=0)  # removes batch dim
    for z in range(zdim):
        z_slice = sparse.select(dim=0, index=z)  # Now 2d
        rows, cols = z_slice.coalesce().indices().tolist()

        non_zero_tiles = {
            (rows[i] // TILE_DIM, cols[i] // TILE_DIM) for i in range(len(rows))
        }

        d = dict()
        for k, v in non_zero_tiles:
            if k not in d:
                d[k] = 0
            d[k] += 1

        max_span = max(max_span, max(d.values()))

    return max_span


def up_idx_to_orig_idx_no_align_corners(up_idx, scale_factor):
    return (up_idx + 0.5) / scale_factor - 0.5


def up_idx_to_orig_idx_align_corners(up_idx, original_size, upsample_size):
    up_idx = up_idx.tolist()
    stride = (original_size - 1) / (upsample_size - 1)
    x_ori_list = []
    # append the first coordinate
    x_ori_list.append(0)
    for i in range(1, len(up_idx) - 1):
        x_ori_list.append(0 + i * stride)
    # append the last coordinate
    x_ori_list.append(original_size - 1)
    return torch.tensor(x_ori_list)


def create_nearest_neighbor_upsample_picker_matrix(
    scale_factor, shape, tile_align=False, for_din=False, channel_last=False,
):
    if channel_last:
        if for_din:
            raise RuntimeError("Resize3d is not supported in channel-last format yet")

        rows = torch.arange(shape[-3] * scale_factor[0] * shape[-2] * scale_factor[1])
        cols = []
        for i in range(shape[-3]):
            col = (
                torch.arange(shape[-2]).repeat_interleave(scale_factor[0]).repeat(scale_factor[1])
                + i * (shape[-2])
            )
            cols.append(col)

        cols = torch.concat(cols)

        sparse_r = rows.shape[0]
        sparse_c = shape[-2] * shape[-3]
        if tile_align:
            sparse_r = align_up_tile(sparse_r)
            sparse_c = align_up_tile(sparse_c)

        return torch.sparse_coo_tensor(
            [rows.tolist(), cols.tolist()], torch.ones(cols.shape[0]), (sparse_r, sparse_c)
        )
    else:
        if for_din:
            rows = torch.arange(shape[-3] * scale_factor[2] * shape[-4])
            #cols = torch.arange(shape[-3]).repeat_interleave(scale_factor)
            cols = []
            for i in range(shape[-4]):
                col = (
                    torch.arange(shape[-3]).repeat_interleave(scale_factor[2])
                    + i * shape[-3]
                )
                cols.append(col)
            cols = torch.concat(cols)
            sparse_r = rows.shape[0]
            sparse_c = shape[-3] * shape[-4]
        else:
            rows = torch.arange(shape[-2] * scale_factor[0] * shape[-1] * scale_factor[1])
            cols = []
            for i in range(shape[-2]):
                col = (
                    torch.arange(shape[-1]).repeat_interleave(scale_factor[0]).repeat(scale_factor[1])
                    + i * shape[-1]
                )
                cols.append(col)
            cols = torch.concat(cols)
            sparse_r = rows.shape[0]
            sparse_c = shape[-1] * shape[-2]

        if tile_align:
            sparse_r = align_up_tile(sparse_r)
            sparse_c = align_up_tile(sparse_c)

        return torch.sparse_coo_tensor(
            [rows.tolist(), cols.tolist()], torch.ones(cols.shape[0]), (sparse_r, sparse_c)
        )

def create_nearest_neighbor_downsample_picker_matrix(
    scale_factor, shape, tile_align=False,
):
    cols = torch.arange(shape[-2] // scale_factor)*scale_factor
    rows = cols // scale_factor
    sparse_r = cols.shape[0]
    sparse_c = shape[-2]
    if tile_align:
        sparse_r = align_up_tile(sparse_r)
        sparse_c = align_up_tile(sparse_c)

    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()], torch.ones(cols.shape[0]), (sparse_r, sparse_c)
    )


def create_bilinear_upsample_picker_matrix(
    scale_factor, shape, align_corners=False, tile_align=False, channel_last=False, split_idx=0, split_factor=1,
):
    r = shape[-2]
    c = shape[-1]

    if channel_last:
        r = shape[-3]
        c = shape[-2]

    # Final dident shape
    num_cols = r * c
    num_rows = num_cols * scale_factor[0] * scale_factor[1]

    upsample_c_idx = torch.arange(0, c * scale_factor[0])
    upsample_r_idx = torch.arange(0, r * scale_factor[1])
    if align_corners:
        upsample_c_idx_adjusted = up_idx_to_orig_idx_align_corners(
            upsample_c_idx, c, c * scale_factor[0]
        )
        upsample_r_idx_adjusted = up_idx_to_orig_idx_align_corners(
            upsample_r_idx, r, r * scale_factor[1]
        )
    else:
        upsample_c_idx_adjusted = up_idx_to_orig_idx_no_align_corners(
            upsample_c_idx, scale_factor[0]
        )
        upsample_r_idx_adjusted = up_idx_to_orig_idx_no_align_corners(
            upsample_r_idx, scale_factor[1]
        )

    # Clip index between 0 and c
    upsample_c_idx_adjusted = torch.clip(upsample_c_idx_adjusted, min=0, max=c - 1)
    upsample_r_idx_adjusted = torch.clip(upsample_r_idx_adjusted, min=0, max=r - 1)

    inp_idx_to_weight_c = {}
    inp_idx_to_weight_r = {}

    # Compute and store idx heat map
    # Key: int
    #   idx in a given row
    # Value: list
    #   The portion that act[idx] contribute to the upsampled output
    #   List length is the upsampled length
    for j in range(c):
        inp_idx = j

        # Step 1: normalize to around the current idx
        normalized_idx = upsample_c_idx_adjusted - inp_idx
        # Step 2: take abs around current idx
        abs_idx = torch.abs(normalized_idx)
        # Step 3: calculate contribution to each upsampled index
        inverted_idx = 1 - abs_idx
        # Step 4: contribution must be between [0,1]
        clipped_idx = torch.where(inverted_idx < 0, torch.zeros(inverted_idx.shape), inverted_idx).to_sparse()

        inp_idx_to_weight_c[inp_idx] = clipped_idx

    for i in range(r):
        inp_idx = i

        # Step 1: normalize to around the current idx
        normalized_idx = upsample_r_idx_adjusted - inp_idx
        # Step 2: take abs around current idx
        abs_idx = torch.abs(normalized_idx)
        # Step 3: calculate contribution to each upsampled index
        inverted_idx = 1 - abs_idx
        # Step 4: contribution must be between [0,1]
        clipped_idx = torch.where(inverted_idx < 0, torch.zeros(inverted_idx.shape), inverted_idx).to_sparse()

        inp_idx_to_weight_r[inp_idx] = clipped_idx

    # Go through each cell in input activation and build up contribution depending on 
    # row and column. 
    baseline = torch.stack([inp_idx_to_weight_c[i % c].unsqueeze(0) for i in range(num_cols)], dim=0)
    idx_elements = torch.stack([inp_idx_to_weight_r[i // c].unsqueeze(-1) for i in range(num_cols)], dim=0)
    indices = torch.bmm(idx_elements, baseline.to_dense()).view(num_cols, num_rows).transpose(-2, -1)
    if split_factor > 1:
        assert num_rows % split_factor == 0
        chunk = num_rows // split_factor
        indices = indices[(split_idx*chunk):((split_idx+1)*chunk)]
    return indices.to_sparse()

def create_conv2d_transpose_weight_dident(kH, kW, tile_align=False):
    rows = torch.arange(kH * kW)
    cols = torch.flip(rows, dims=[0])

    sparse_r = rows.shape[0]
    sparse_c = cols.shape[0]

    if tile_align:
        sparse_r = align_up_tile(sparse_r)
        sparse_c = align_up_tile(sparse_c)

    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()], torch.ones(cols.shape[0]), (sparse_r, sparse_c)
    )


def create_conv2d_transpose_input_act_dident(y, x, stride, tile_align=False):
    cols = torch.arange(start=1, end=y * x + 1).view(y, x)

    # Stride is used to dilate input activation
    if stride != 1:
        gap_cols = torch.zeros((y * stride, x * stride))
        gap_cols[::stride, ::stride] = cols
        cols = gap_cols

    cols = cols.reshape(-1)
    rows = torch.arange(cols.shape[0])
    rows = rows.index_select(0, cols.nonzero().flatten())
    cols = cols.index_select(0, cols.nonzero().flatten())
    cols -= 1

    sparse_r = (y * stride) * (x * stride)
    sparse_c = y * x
    if tile_align:
        sparse_r = align_up_tile(sparse_r)
        sparse_c = align_up_tile(sparse_c)
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (sparse_r, sparse_c),
        dtype=torch.float32,
    )


def create_eye_sparse_picker_matrix(r, tile_align=False):
    eye = torch.arange(r)
    sparse_r = eye.shape[0]
    sparse_c = eye.shape[0]
    if tile_align:
        sparse_r = align_up_tile(sparse_r)
        sparse_c = align_up_tile(sparse_c)
    return torch.sparse_coo_tensor(
        [eye.tolist(), eye.tolist()], torch.ones(eye.shape[0]), (sparse_r, sparse_c)
    )


def create_all_around_padding_picker_matrix(shape, padding, channel_last=False, tile_align=False):

    assert len(padding) == 4
    padding_left, padding_right, padding_top, padding_bottom, = padding

    if channel_last:
        r = shape[-3]
        c = shape[-2]
    else:
        r = shape[-2]
        c = shape[-1]

    c_index = torch.arange(r * c)
    r_index = (torch.arange(r * c) + 1).tolist()
    r_index_rows = [r_index[x : x + c] for x in range(0, len(r_index), c)]

    for i, row in enumerate(r_index_rows):
        r_index_rows[i] = [0] * padding_left + row + [0] * padding_right
    r_index = [item for row in r_index_rows for item in row]

    r_index_final = torch.Tensor(
        [0] * (c + padding_left + padding_right) * padding_top
        + r_index
        + [0] * (c + padding_left + padding_right) * padding_bottom
    )

    r_index = torch.nonzero(r_index_final).flatten().tolist()
    num_cols = r * c
    num_rows = (r + padding_top + padding_bottom) * (c + padding_left + padding_right)
    if tile_align:
        num_rows = align_up_tile(num_rows)
        num_cols = align_up_tile(num_cols)

    return torch.sparse_coo_tensor(
        [
            r_index,
            c_index.tolist(),
        ],
        torch.ones(r * c),
        (num_rows, num_cols),
    )


def vslice_cols(t: torch.Tensor, split_factor: int):
    assert t.is_sparse, "this works only for sparse tensors"
    assert len(t.shape) == 2

    t = t.coalesce()
    assert t.shape[0] % split_factor == 0, "invalid vslice split factor"

    ret = [[]] * split_factor
    slice_height = t.shape[0] // split_factor
    rows, cols = t.indices().tolist()
    vals = t.values().tolist()
    for idx in range(len(rows)):
        slice_idx = rows[idx] // slice_height
        ret[slice_idx].append(cols[idx])

    return ret


def vslice(t: torch.Tensor, split_factor: int, pack=False):
    assert t.is_sparse, "this works only for sparse tensors"
    assert len(t.shape) == 2

    t = t.coalesce()

    np.empty((split_factor, 3))
    ret = [[[] for _ in range(3)] for __ in range(split_factor)]
    assert t.shape[0] % split_factor == 0, "invalid vslice split factor"
    slice_height = t.shape[0] // split_factor

    rows, cols = t.indices().tolist()
    vals = t.values().tolist()
    for idx in range(len(rows)):
        slice_idx = rows[idx] // slice_height
        ret[slice_idx][0].append(rows[idx] % slice_height)
        ret[slice_idx][1].append(cols[idx])
        ret[slice_idx][2].append(vals[idx])

    ret = [torch.sparse_coo_tensor(indices=[r[0], r[1]], values=r[2], size=(t.shape[0] // split_factor, t.shape[1]), dtype=t.dtype) for r in ret]

    if pack:
        ret = torch.stack(ret, dim=0)

    return ret


def hslice(t: torch.Tensor, split_factor: int, pack=False):
    assert t.is_sparse, "this works only for sparse tensors"
    assert len(t.shape) == 2

    if not t.is_coalesced():
        t = t.coalesce()

    np.empty((split_factor, 3))
    ret = [[[] for _ in range(3)] for __ in range(split_factor)]
    assert t.shape[1] % split_factor == 0, "invalid hslice split factor"
    slice_width = t.shape[1] // split_factor

    rows, cols = t.indices().tolist()
    vals = t.values().tolist()
    for idx in range(len(rows)):
        slice_idx = cols[idx] // slice_width
        ret[slice_idx][0].append(rows[idx])
        ret[slice_idx][1].append(cols[idx] % slice_width)
        ret[slice_idx][2].append(vals[idx])

    ret = [torch.sparse_coo_tensor(indices=[r[0], r[1]], values=r[2], size=(t.shape[0], t.shape[1] // split_factor), dtype=t.dtype) for r in ret]

    if pack:
        ret = torch.stack(ret, dim=0)

    return ret


def hstack(t: torch.Tensor, stack_factor: int):
    assert t.is_sparse, "this works only for sparse tensors"
    assert len(t.shape) == 4

    if not t.is_coalesced():
        t = t.coalesce()

    assert t.shape[1] % stack_factor == 0, "invalid hstack stack factor"
    slices_count = t.shape[1] // stack_factor
    ret = [[] for _ in range(5)]

    ws, zs, rows, cols = t.indices().tolist()
    vals = t.values().tolist()
    for idx in range(len(rows)):
        slice_idx = zs[idx] // stack_factor
        ret[0].append(ws[idx])
        ret[1].append(slice_idx)
        ret[2].append(rows[idx])
        ret[3].append(cols[idx] + (zs[idx] % stack_factor) * t.shape[-1])
        ret[4].append(vals[idx])

    ret = torch.sparse_coo_tensor(indices=[ret[0], ret[1], ret[2], ret[3]], values=ret[4], size=(t.shape[0], slices_count, t.shape[2], t.shape[3] * stack_factor), dtype=t.dtype)
    return ret


def vslice_interleave(sparse, grid_r, t_factor, bcast_factor):
    vsliced = vslice(sparse, grid_r * t_factor * bcast_factor)
    slices = []
    for t in range(t_factor):
        b_slices = []
        for b in range(bcast_factor):
            for r in range(grid_r):
                i = b * t_factor * grid_r + t * grid_r + r
                b_slices.append(vsliced[i])
        for r in range(grid_r):
            i = r * bcast_factor
            slices.append(torch.cat(b_slices[i:(i+bcast_factor)]))
    return slices


def interleave_tiles(pickers: "list[torch.Tensor]"):
    rows, cols, vals = [], [], []

    zdim = len(pickers)

    for z, picker in enumerate(pickers):
        picker = picker.coalesce()
        curr_rows, curr_cols = picker.indices().tolist()
        # curr_cols = [zdim * c + z for c in curr_cols]  # interleaves scalars...
        curr_cols = [(c // 32) * zdim * 32 + z * 32 + c % 32 for c in curr_cols]
        rows.extend(curr_rows)
        cols.extend(curr_cols)
        vals.extend(picker.values().tolist())

        assert picker.shape == pickers[0].shape

    return torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=vals,
        size=(pickers[0].shape[0], pickers[0].shape[1] * zdim),
        dtype=pickers[0].dtype
    ).coalesce()


def create_sparse_buda(sparse, bcast_factor=1, tile_align=True) -> SparseBUDA:
    while len(sparse.shape) < 4:
        sparse = sparse.unsqueeze(0)
    w, zdim, y, x = sparse.shape
    assert w == 1

    if tile_align:
        pad_y = align_up_tile(y) - y
        pad_x = align_up_tile(x) - x
        if pad_y + pad_x > 0:
            y += pad_y
            x += pad_x
            # torch.nn.functional.pad() doesn't work for sparse so recreating tensor
            sparse = torch.sparse_coo_tensor(
                indices=sparse.coalesce().indices(),
                values=sparse.coalesce().values(),
                size=(w, zdim, y, x),
                dtype=sparse.dtype,
            )

    sparse_zs = []
    for z in range(zdim):
        zslice = sparse[0][z].coalesce()
        rows, cols = zslice.indices()
        vals = zslice.values()
        assert len(zslice.shape) == 2
        sparse_zs.append(
            SparseCOO(rows.tolist(), cols.tolist(), vals.tolist(), list(zslice.shape))
        )

    return compress_sparse_tensor_and_strip_info(sparse_zs, bcast_factor)


def shapeify_sparse_tiles_and_encodings(sparse, encodings, grid_r):
    sparse_tensor = torch.tensor(sparse).view(1, 1, -1, TILE_DIM)  # TODO: Do we want a specific dtype?
    sparse_tensor = sparse_tensor.reshape(-1, TILE_DIM, TILE_DIM).transpose(0, 1).reshape(TILE_DIM, -1).hsplit(grid_r)
    sparse_tensor = torch.cat(sparse_tensor, dim=0).view(1, 1, TILE_DIM * grid_r, -1)

    # Encodings behave differently... all the encoding scalars should be laid out in memory in row-major order, but
    # the resulting tensor should be shaped as the netlist would see it - BE will read scalar by scalar and fill up tile
    # by tile
    encodings_tensor = torch.tensor(encodings, dtype=torch.int32).view(1, 1, grid_r * TILE_DIM, -1)

    return sparse_tensor, encodings_tensor


def compress_buda_picker(sparse, strip_info=True, tile_align=True):
    while len(sparse.shape) < 4:
        sparse = sparse.unsqueeze(0)
    w, zdim, y, x = sparse.shape
    assert w == 1

    sparse_zs = []
    for z in range(zdim):
        zslice = sparse[0][z].coalesce()
        rows, cols = zslice.indices()
        vals = zslice.values()
        assert len(zslice.shape) == 2
        sparse_zs.append(
            SparseCOO(rows.tolist(), cols.tolist(), vals.tolist(), list(zslice.shape))
        )

    assert strip_info == True, "removed compress_sparse_tensor, oops, ping svuckovic"  # TODO: Check with Nick if we can remove this code
    sparse_buda = (
        compress_sparse_tensor_and_strip_info(sparse_zs, 1)
        if strip_info
        else compress_sparse_tensor(sparse_zs)
    )

    # TODO: Code below needs fixing to account for grid_r > 1
    unique_tiles = torch.tensor(sparse_buda.tiles).view(1, 1, -1, TILE_DIM)
    unique_tiles = unique_tiles.reshape(-1, TILE_DIM, TILE_DIM).transpose(0, 1).reshape(1, 1, TILE_DIM, -1)
    indices_len = len(sparse_buda.indices)
    pad_right = align_up(indices_len, TILE_DIM * TILE_DIM) - indices_len
    indices = torch.tensor(sparse_buda.indices, dtype=torch.int32)
    indices = torch.nn.functional.pad(indices, (0, pad_right))
    indices = indices.view(1, 1, TILE_DIM, -1)
    assert indices.shape[-2] % TILE_DIM == 0

    logger.trace("Sparse BUDA Tensor:")
    logger.trace("  Unique Tiles: {}", unique_tiles.shape[-1] // TILE_DIM)
    logger.trace("  Index Tiles: {}", indices.shape[-1] // TILE_DIM)

    # return unique_tiles, indices, indices_len, sparse_buda.sparse_tile_ptr_bits
    return unique_tiles, indices, indices_len, sparse_buda


def num_sparse_tiles_in_strip(sparse, verbose=False):
    rt = align_up_tile(sparse.shape[-2]) // TILE_DIM
    ct = align_up_tile(sparse.shape[-1]) // TILE_DIM
    rows, cols = sparse.coalesce().indices().tolist()
    out = torch.zeros((rt, ct), dtype=torch.int32)
    for r, c in zip(rows, cols):
        out[r // TILE_DIM, c // TILE_DIM] = 1

    if verbose:
        for r in range(rt):
            for c in range(ct):
                print("." if out[r, c].item() == 0 else "1", end='')
            print("")
    return out.sum(dim=-2)


def get_u_kts(k):
    u_kts = []
    for f in range(1, k):
        if f > 32:
            break
        if (k % f) == 0:
            u_kts.append(f)
    return u_kts


def calculate_total_sparse_tile_util(slices, grid_r, ts, bcast_factor, verbose=False):
    if verbose:
        print("="*32)
        print(f"= Grid_r[{grid_r}] T[{ts}] Bfactor[{bcast_factor}]")
        print("="*32)
    u_kts = get_u_kts(align_up_tile(slices[0].shape[-1]) // TILE_DIM)
    total_util = {}
    for u_kt in u_kts:
        total_util[u_kt] = 0.0
    for t in range(ts):
        col_counts = []
        for gr in range(grid_r):
            if verbose:
                print(f"t[{t}] grid_r[{gr}]")
            r_slice = slices[t * grid_r + gr]
            col_counts.append(num_sparse_tiles_in_strip(r_slice, verbose=verbose))

        col_counts = torch.stack(col_counts)
        for u_kt in u_kts:
            ucol_counts = col_counts.reshape(grid_r, -1, u_kt)
            ucol_counts = ucol_counts.sum(dim=-1)
            max_counts = ucol_counts.max(dim=0)[0]
            max_n = torch.count_nonzero(max_counts)
            util = ucol_counts / max_counts
            util = torch.nan_to_num(util, nan=0.0)
            util = (util.sum(dim=-1) / max_n).mean()
            total_util[u_kt] += util
    for u_kt in u_kts:
        util = total_util[u_kt] / ts
        print(f"> Grid_r[{grid_r:2}] T[{ts:2}] Bfactor[{bcast_factor:2}] u_kt[{u_kt:2}] -> total_util [{util:0.4}] speedup[{util * grid_r:0.4}]")


def can_conv2d_prestride(act_shape, weight_shape, stride, dilation, groups, padding, channel_last, graph_input):
    if not graph_input:
        return False

    stride_height, stride_width = stride
    cout, cin, kH, kW = weight_shape

    # Only support square stride
    # Non-square strides do work in terms of math, but backend would need to add support as well
    # tenstorrent/budabackend#1519
    if stride_height != stride_width:
        return False

    # stride <= 1 can't be prestrided
    if stride_height <= 1 and stride_width <= 1:
        return False

    # Skip for now
    if channel_last:
        return False

    # Dilation unsupported
    if dilation > 1:
        return False

    if groups > 1:
        return False

    return True


def does_prestriding_improve_perf(act_shape, weights_shape, ps_weights, stride):
    assert len(act_shape) == 4
    assert len(weights_shape) == 4
    assert len(ps_weights) == 4

    w, act_cin, y, x = act_shape
    cout, cin, kH, kW = weights_shape
    assert act_cin == cin, f"{act_cin} {cin}"

    orig_m = align_up_tile(y * x) // TILE_DIM
    orig_k = align_up_tile(cin) * kH * kW // TILE_DIM
    orig_n = align_up_tile(cout) // TILE_DIM
    # m * k * n
    orig_total = orig_m * orig_k * orig_n

    ps_cout, ps_cin, ps_kH, ps_kW = ps_weights
    assert ps_cout == cout
    ps_y = (y + stride - 1) // stride
    ps_x = (x + stride - 1) // stride
    ps_m = align_up_tile(ps_y * ps_x) // TILE_DIM
    ps_k = align_up_tile(ps_cin) * ps_kH * ps_kW // TILE_DIM
    ps_n = align_up_tile(ps_cout) // TILE_DIM
    # m * k * n
    ps_total = ps_m * ps_k * ps_n
    return ps_total < orig_total


def can_fracture_conv_at_op_level(attr, dc, inputs):
    # Can't fracture if disabled (unless manual op-level override)
    if bool(int(os.environ.get("PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE", "0"))):
        return False

    return inputs[1].shape.r > 1


def does_fracturing_conv_at_op_level_improve_perf(attr, dc, inputs):
    stride = [attr[0], attr[1],]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    channel_last = attr[-1]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    if channel_last:
        _, yout, xout, _ = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]
        w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
    else:
        w, cin, y, x = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        _, _, yout, xout = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]

    cout, _, kH, kW = (weights.shape.w, weights.shape.z, weights.shape.r, weights.shape.c)

    # Ideally, balancer would make the fracturing decision, until then we can try to guesstimate here
    # There's no great way to do it though, many factors can affect the utilization, dataflow bw, etc...
    #
    # Given a couple examples, looks like we need 140-170 cycles/tile for sparse matmul - let's favor slower cases, so ~170 cycles/tile
    # We often encounter some dataflow issues that slow this down, so let's add a healthy margin of 100, so we're at ~270 cycles/tile
    # In Ribbon2, we aim for ~95k cycles per op, which means we can do 95k/250 ~= 350 tiles per core in height
    # We don't know what parallelizations the op will have, so I'll randomly pick 4 cores as something we should usually be able to do, not more
    #
    # So if we have more than 4 cores * 350 tiles = 1400 tiles, let's fracture
    frac_threshold = 1400
    total_rt = align_up_tile(yout * xout) // 32 * kH * kW  # total row tiles produced by sparse mm

    return total_rt > frac_threshold


def should_fracture_conv_at_op_level(attr, dc, inputs):
    # We currently can't fracture when training, due to `index` op not having bw op defined
    # tenstorrent/pybuda#972
    if dc.is_training_enabled():
        return False

    # Check if fracturing is forced
    if bool(int(os.environ.get("PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE", "0"))) and inputs[1].shape.r > 1:
        logger.warning("Environment variable \"PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE\" will be deprecated soon, please do NOT use.")
        return True

    # Check if user overriden
    compiler_cfg = dc.get_compiler_cfg()
    if dc.node_name() in compiler_cfg.conv_multi_op_fracture_factor_override:
        # Currently we only support fracturing by kernel height but we could be more flexible here if needed!
        kH = inputs[1].shape.r
        if compiler_cfg.conv_multi_op_fracture_factor_override[dc.node_name()] == -1:
            compiler_cfg.conv_multi_op_fracture_factor_override[dc.node_name()] = kH
        assert compiler_cfg.conv_multi_op_fracture_factor_override[dc.node_name()] == kH, "Invalid multi_op_fracture_factor used!"
        return True

    return can_fracture_conv_at_op_level(attr, dc, inputs) and does_fracturing_conv_at_op_level_improve_perf(attr, dc, inputs)


def visualize_sparse(sparses, file_path, grid_r=1, ts=1):
    import matplotlib
    import matplotlib.pyplot as plt
    import os

    if type(sparses) is not list:
        sparses = [sparses]

    assert len(sparses) == grid_r * ts

    scale = 10
    f = plt.figure(figsize=(ts * scale, grid_r * scale))
    plt.suptitle(file_path)

    for g_r in range(grid_r):
        for t in range(ts):
            idx = g_r * ts + t
            sparse = sparses[idx]

            if type(sparse) is torch.Tensor:
                rows, cols = sparse.coalesce().indices().tolist()
            elif type(sparse) is SparseCOO:
                rows, cols = (sparse.rows, sparse.cols)
            else:
                raise TypeError(f"Unsupported sparse tensor type: {type(sparse)}")

            rt = align_up_tile(sparse.shape[-2]) // TILE_DIM
            ct = align_up_tile(sparse.shape[-1]) // TILE_DIM

            tilized_rows = []
            tilized_cols = []
            for r, c in zip(rows, cols):
                if len(tilized_rows) == 0 or (tilized_rows[-1] != (r // TILE_DIM)) or (tilized_cols[-1] != (c // TILE_DIM)):
                    tilized_rows.append(r // TILE_DIM)
                    tilized_cols.append(c // TILE_DIM)

            sub = plt.subplot(grid_r, ts, idx + 1)
            sub.set_title(f"grid_r={g_r} t={t}")
            sub.set_xlim([0, ct])
            sub.set_ylim([0, rt])
            sub.invert_yaxis()
            sub.scatter(tilized_cols, tilized_rows, s=3, c="b", marker="s")

    plt.savefig(file_path)
    plt.clf()


def pad_sparse_coo_tensor(sparse_tensor, new_shape):
    
    assert sparse_tensor.shape[-2] <= new_shape[-2]
    assert sparse_tensor.shape[-1] <= new_shape[-1]
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()

    result = torch.sparse_coo_tensor(indices, values, new_shape)
    
    return result


def create_pad_replicate_sparse_picker(r, c, left, right, top, bottom):
    new_shape = (r+top+bottom) * (c+left+right)
    orig_shape = r*c

    rows = torch.arange(new_shape).tolist()
    cols = []

    # dim=-1 padding
    last_dim_index = []
    for _ in range(left):
        last_dim_index.append(0)
    last_dim_index.extend(torch.arange(c).tolist())
    for _ in range(right):
        last_dim_index.append(c-1)

    # dim=-2 padding
    for _ in range(top):
        cols.extend(last_dim_index)
    [cols.extend((torch.tensor(last_dim_index) + offset*c).tolist()) for offset in range(r)]
    for _ in range(bottom):
        cols.extend((torch.tensor(last_dim_index) + (r-1)*c).tolist())

    spm = torch.sparse_coo_tensor((rows, cols), torch.ones(new_shape), (new_shape, orig_shape))
    return spm

def create_pad_reflect_sparse_picker(r, c, left, right, top, bottom):
    new_shape = (r+top+bottom) * (c+left+right)
    orig_shape = r*c

    rows = torch.arange(new_shape).tolist()
    cols = []

    horizontal_indices = [left - cidx for cidx in range(left)] + torch.arange(c).tolist() + [c - cidx - 2 for cidx in range(right)]
    vertical_indices = [top - ridx for ridx in range(top)] + torch.arange(r).tolist() + [r - ridx - 2 for ridx in range(bottom)]

    for vertical_index in vertical_indices:
        
        vectical_offset = vertical_index * c
        last_dim_index = [x + vectical_offset for x in horizontal_indices]
        cols.extend(last_dim_index)

    spm = torch.sparse_coo_tensor((rows, cols), torch.ones(new_shape), (new_shape, orig_shape))
    return spm


def pad_tensor_zeros(shape):
    return torch.zeros(shape)


def pad_tensor_identity(shape, pad):
    size = shape[-2]
    shape[-2] = size + pad
    shape[-1] = size
    tensor = torch.concat((torch.eye(size), torch.zeros(pad, size)), dim=-2)
    return torch.broadcast_to(tensor, shape)


def pad_tensor_identity_sparse(size, pad):
    return pad_tensor_identity(size, pad).to_sparse()


def pad_tensor_identity_transposed(size, pad):
    return torch.transpose(torch.concat((torch.eye(size), torch.zeros(pad, size)), dim=-2), -1, -2)


def pad_tensor_identity_sparse_transposed(size, pad):
    return pad_tensor_identity_transposed(size, pad).to_sparse()


def conv2d_out_shape(type, attr, ops):
    assert len(ops) <= 3, "Conv2d should have three inputs"
    # assert len(attr) == 10, f"Conv2d invalid num attributes: {len(attr)}"

    activations = ops[0]
    weights = ops[1]
    kernel_size = [weights[2], weights[3]]
    stride = [attr[0], attr[1],]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    is_convtranspose2d = attr[8]  # True if decomposed from convtranspose2d
    channel_last = attr[-1]

    if channel_last == 1:
        in_y = activations[1]
        in_x = activations[2]
    else:
        in_y = activations[2]
        in_x = activations[3]

    if type == "conv2d":
        y, x = calculate_conv2d_output_dimensions(
            in_y, in_x, kernel_size, stride, padding, dilation
        )

        # TODO: the existence of this `if` block is a but confusing, should be fixed once this proposal is implemented:
        # tenstorrent/pybuda#1761
        if is_convtranspose2d:
            # if transposed conv, the output is calculated by `calculate_conv2d_transpose_output_dimensions()`
            # however, we can't call this function on conv2d, as some attributes have been changed to fit the style of
            # conv2d (e.g. padding) - it would produce wrong numbers
            yout_transpose = attr[9]
            xout_transpose = attr[10]
            y = yout_transpose
            x = xout_transpose

        if channel_last == 1:
            return (activations[0], y, x, weights[0]), []
        else:
            return (activations[0], weights[0], y, x), []
    elif type == "conv2d_transpose":
        assert dilation == 1, "Currently only support dilation = 1"
        assert all([p == padding[0] for p in padding]), "ConvTranspose2d only supports same padding on all sides"
        assert all([s == stride[0] for s in stride]), "ConvTranspose2d only supports same strides"
        stride = stride[0]

        y, x = calculate_conv2d_transpose_output_dimensions(
            in_y, in_x, (weights[2], weights[3]), stride, padding, dilation=dilation
        )

        if channel_last == 1:
            return (activations[0], y, x, weights[1] * groups), []
        else:
            return (activations[0], weights[1] * groups, y, x), []

def conv3d_out_shape(type, attr, ops):
    assert len(ops) <= 3, "Conv3d should have three inputs"
    assert len(attr) == 12, f"Conv3d invalid num attributes: {len(attr)}"

    activations = ops[0]
    weights = ops[1]
    kernel_size = [weights[2], weights[3], weights[4]]
    stride = [attr[0], attr[1], attr[2]]
    dilation = attr[3]
    groups = attr[4]
    padding = [attr[5], attr[6], attr[7], attr[8], attr[9], attr[10]]

    if attr[-1] == 1:
        # Channel last
        in_z = activations[1]
        in_y = activations[2]
        in_x = activations[3]
    else:
        in_z = activations[2]
        in_y = activations[3]
        in_x = activations[4]

    if type == "conv3d":
        z, y, x = calculate_conv3d_output_dimensions(
            in_z, in_y, in_x, kernel_size, stride, padding, dilation
        )
        if attr[-1] == 1:
            return (activations[0], z, y, x, weights[0]), []
        else:
            return (activations[0], weights[0], z, y, x), []

def pad_to_fit_dile_dims(tensor):
    """
    Pads the input tensor to ensure its height and width are divisible by TILE_DIM.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor of shape (B, C, H, W)

    Returns
    -------
    torch.Tensor - The padded tensor with shape (B, C, H + pad_height, W + pad_width).
    """

    height, width = tensor.shape[-2:]

    pad_height = (TILE_DIM - height % TILE_DIM) % TILE_DIM
    pad_width = (TILE_DIM - width % TILE_DIM) % TILE_DIM

    return torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

def count_unique_tiles(tensor):
    """
    Counts the number of unique tiles in the tensor.

    Parameters:
    tensor: (torch.Tensor)
        The input tensor of shape (B, C, H, W)

    Returns:
    int: The number of unique tiles in the tensor.
    """

    padded_tensor = pad_to_fit_dile_dims(tensor)

    height, width = padded_tensor.shape[-2:]
    assert height % TILE_DIM == 0 and width % TILE_DIM == 0, f"Dimensions must be divisible by {TILE_DIM}"

    # Shape: (B, C, H_tiles, W_tiles, TILE_DIM, TILE_DIM)
    unfolded = padded_tensor.unfold(2, TILE_DIM, TILE_DIM).unfold(3, TILE_DIM, TILE_DIM)

    # Shape: (B * C * H_tiles * W_tiles, TILE_DIM, TILE_DIM)
    tiles = unfolded.reshape(-1, TILE_DIM, TILE_DIM)

    unique_tiles = torch.unique(tiles, dim=0)
    return unique_tiles.size(0)

def check_sparse_tensor_unique_tiles(tensor):
    """
    Checks if the number of unique tiles in the sparse tensor does not exceed the maximum allowed limit.

    Parameters:
    tensor: (torch.Tensor)
        The input tensor of shape (B, C, H, W)

    Returns:
    bool: True if the number of unique tiles is less than or equal to max_unique_tiles, False otherwise.
    """
    
    max_unique_tiles = 4096

    return count_unique_tiles(tensor) <= max_unique_tiles