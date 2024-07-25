# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import ast
import torch

from pybuda._C.graph import NodeType
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile, round_up_div, clamp
from pybuda import Tensor
from pybuda.config import _get_global_compiler_config
from .transpose import TransposeTM
from .buffer import Buffer

from ..common import to_torch_operands
from ..sparse_utils import (
    conv2d_out_shape,
    conv3d_out_shape,
    create_conv2d_sparse_picker_matrix,
    create_conv2d_transpose_weight_dident,
    create_conv2d_transpose_input_act_dident,
    interleave_tiles,
    does_prestriding_improve_perf,
    can_conv2d_prestride,
    calculate_conv2d_prestride_weights_and_padding,
    should_fracture_conv_at_op_level,
)


def decomp_group_convolution(type, attr, dc, inputs):
    stride = [attr[0], attr[1],]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    in_channels = activations.shape.z

    axis = -3 # Z dim
    slice_per_group = in_channels // groups
    output_channel_per_group = weights.shape.w // groups

    # Indexing on W is not supported, transpose to Z
    weights = dc.op(TransposeTM.create(0, 1), [weights])
    parallel_convs = []
    for i in range(groups):
        start_idx = i * slice_per_group
        end_idx = (i + 1) * slice_per_group
        slice_act = dc.op("index", [activations], (axis, start_idx, end_idx, 1))
        start_idx = i * output_channel_per_group
        end_idx = (i + 1) * output_channel_per_group
        slice_weight = dc.op("index", [weights], (axis, start_idx, end_idx, 1))
        # Transpose back to OIHW after indexing on Z
        slice_weight = dc.op(TransposeTM.create(0, 1), [slice_weight])
        new_conv2d = dc.op("conv2d", [slice_act, slice_weight], (stride[0], stride[1], dilation, 1, padding[0], padding[1], padding[2], padding[3]))
        parallel_convs.append(new_conv2d)

    result = dc.op("concatenate", parallel_convs, (axis,))
    if bias != None:
        bias_shape = bias.shape.as_list() + [1, 1]
        bias = dc.op("reshape", [bias], bias_shape)
        result = dc.op("add", [result, bias])
    dc.fuse(result)


def should_prestride(attr, dc, inputs):
    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    act_shape = activations.shape.as_list()
    weight_shape = weights.shape.as_list()
    stride = (attr[0], attr[1])
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    channel_last = attr[-1]
    graph_input = activations.node_type == NodeType.kInput

    if not can_conv2d_prestride(
            act_shape,
            weight_shape,
            stride,
            dilation,
            groups,
            padding,
            channel_last,
            graph_input,
        ):
        return False

    w, cin, y, x = act_shape
    ps_weights, ps_padding = calculate_conv2d_prestride_weights_and_padding(weight_shape, y, x, stride[0], padding)
    return does_prestriding_improve_perf(act_shape, weight_shape, ps_weights, stride[0])


def decompose_conv2d_prestride(attr, dc, inputs):
    stride_height, stride_width = attr[0], attr[1]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    channel_last = attr[-1]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    cout, _, kH, kW = (weights.shape.w, weights.shape.z, weights.shape.r, weights.shape.c)

    if channel_last:
        assert False, "channel_last not supported in conv prestriding path"

    if channel_last:
        _, yout, xout, _ = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]
        w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
    else:
        w, cin, y, x = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        _, _, yout, xout = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]

    prestrided_activations = dc.op("conv2d_prestride_act", [activations], (stride_height, stride_width, kH, kW, activations.shape[-2], activations.shape[-1]), optimize_hoist=True)
    prestrided_weights = dc.op("conv2d_prestride_weights", [weights], (y, x, stride_height, stride_width, *padding), optimize_hoist=True)

    # Prestride also does reshape/transpose of tensor into "conv space", so undo it here
    # prestrided_activations = dc.op(TransposeTM.create(2, 3), [prestrided_activations])
    prestrided_activations = dc.op("reshape", [prestrided_activations], (w, prestrided_activations.shape.r, round_up_div(y, stride_height), round_up_div(x, stride_width)))

    ps_weights, ps_padding = calculate_conv2d_prestride_weights_and_padding((cout, cin, kH, kW), y, x, stride_width, padding)

    new_conv_attr = list(attr)
    new_conv_attr[0] = 1  # new stride height will be 1
    new_conv_attr[1] = 1  # new stride width will be 1
    new_conv_attr[4] = ps_padding[0]
    new_conv_attr[5] = ps_padding[1]
    new_conv_attr[6] = ps_padding[2]
    new_conv_attr[7] = ps_padding[3]
    prestrided_conv_operands = [prestrided_activations, prestrided_weights]
    if bias:
        prestrided_conv_operands.append(bias)
    prestrided_conv = dc.op("conv2d", prestrided_conv_operands, tuple(new_conv_attr))

    dc.fuse(prestrided_conv)


def decompose_fracture_conv2d_at_op_level(attr, dc, inputs):
    stride_height, stride_width = attr[0], attr[1]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    is_convtranspose2d = attr[8]  # True if decomposed from convtranspose2d
    if is_convtranspose2d:
        yout_transpose = attr[9]
        xout_transpose = attr[10]
        stride_transpose = attr[11]
    channel_last = attr[-1]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    _, _, kH, kW = (weights.shape.w, weights.shape.z, weights.shape.r, weights.shape.c)

    # Fracture convs into multiple ops
    fractured_convs = []
    for curr_kH in range(kH):
        fractured_weights = dc.op("index", [weights], (-2, curr_kH, curr_kH + 1, 1), dont_decompose=True, output_df=weights.output_df)
        fractured_conv_operands = [activations, fractured_weights]
        if curr_kH == 0 and bias:
            fractured_conv_operands.append(bias)

        # Copy attrs, change only top/bottom padding
        fractured_conv_attrs = list(attr)
        fractured_conv_attrs[6] = padding[2] - curr_kH
        fractured_conv_attrs[7] = padding[3] - kH + 1 + curr_kH

        fractured_conv = dc.op("conv2d", fractured_conv_operands, fractured_conv_attrs)
        fractured_convs.append(fractured_conv)

    # Eltwise add all the fractured convs
    while len(fractured_convs) > 1:
        left = fractured_convs.pop(0)
        right = fractured_convs.pop(0)
        result = dc.op("add", [left, right], output_df=left.output_df)
        fractured_convs.append(result)

    dc.fuse(fractured_convs[0])


def transform_weights_for_conv2d(dc, weights, cin, cout, depthwise, groups, kH, kW):
    grouped_conv = groups > 1

    weights = dc.op("reshape", [weights], (1, cout, cin // groups, kH * kW), output_df=weights.output_df)
    weights = dc.op(TransposeTM.create(1, 3, kH * kW), [weights], output_df=weights.output_df)

    if depthwise:
        weights = dc.op("conv2d_depthwise_weights", [weights], output_df=weights.output_df)
    elif grouped_conv:
        weights = dc.op("conv2d_grouped_weights", [weights], (groups, kH * kW, cin // groups, cout, True), output_df=weights.output_df)
        if kH * kW > 1:
            weights = dc.op("hslice", [weights], (kH * kW,), output_df=weights.output_df)
            weights = dc.op("vstack", [weights], (kH * kW,), output_df=weights.output_df)
    else:
        weights = dc.op("pad_tile", [weights], (-2, cin // groups), output_df=weights.output_df)
        weights = dc.op("reshape", [weights], (1, 1, align_up_tile(cin // groups) * kH * kW, cout), output_df=weights.output_df)

    return weights


def rotate_convtranspose2d_weights(dc, weights, cin, cout, depthwise, groups, kH, kW):
    """Rotate weight matrix 180 degrees.

    For convtranspose2d op, the weight matrix needs to be rotated 180 deg. This can be done independently from the conv
    op it gets decomposed into.
    """
    # TODO: A "rotate" op can be added since weights get consteval'd (e.g. 2x torch.rot90())

    # Conv2dTranspose has shape (cin, cout, kH, kW), need to transpose (0,1) first
    # Note: weights for regular conv are (out_channels, in_channels/groups, kH, kW)
    # Note: weights for transpo conv are (in_channels, out_channels/groups, kH, kW)
    weights = dc.op(TransposeTM.create(0, 1), [weights])
    weights = dc.op("reshape", [weights], (1, cout, cin // groups, kH * kW))
    weights = dc.op(TransposeTM.create(2, 3), [weights]) # Transpose weight

    # Create weight dident to rotate last 2 dims by 180 degrees
    # eg. [[1,2] ,[3,4]] -> [[4,3] ,[2,1]]
    if cout > 1:
        weights = dc.op("hstack", [weights], (cout,))
    weight_dident = create_conv2d_transpose_weight_dident(kH, kW, tile_align=False).unsqueeze(0).unsqueeze(0)
    weight_dident_tensor = dc.tensor(weight_dident)
    weights = dc.op("sparse_matmul", [weight_dident_tensor, weights])

    if cout > 1:
        row_after_hslice = weights.shape[-1] // cout
        if row_after_hslice % TILE_DIM != 0:
            orig_w_shape = weights.shape
            weights = dc.op("reshape", [weights], (orig_w_shape[-4], orig_w_shape[-3], orig_w_shape[-2]*cout, row_after_hslice))
            weights = dc.op("pad_tile", [weights], (-1, weights.shape[-1]))
            weights = dc.op("reshape", [weights], (orig_w_shape[-4], orig_w_shape[-3], orig_w_shape[-2], align_up_tile(row_after_hslice)*cout))
            weights = dc.op("hslice", [weights], (cout,))
            weights = dc.op("narrow", [weights], (-1, 0, row_after_hslice, weights.shape[-1]))
        else:
            weights = dc.op("hslice", [weights], (cout,))
    weights = dc.op(TransposeTM.create(2, 3), [weights]) # Transpose weight
    # Reshape into conv2d weight shape
    if depthwise:
        weights = dc.op("reshape", [weights], (cin, cout // groups, kH, kW))
    else:
        weights = dc.op("reshape", [weights], (cout // groups, cin, kH, kW))

    return weights


def decompose_conv2d_sparse_first(attr, dc, inputs):
    stride = [attr[0], attr[1],]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    is_convtranspose2d = attr[8]  # True if decomposed from convtranspose2d
    yout_transpose = attr[9]
    xout_transpose = attr[10]
    stride_transpose = attr[11]
    channel_last = attr[-1]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    cout, _, kH, kW = (weights.shape.w, weights.shape.z, weights.shape.r, weights.shape.c)

    if channel_last:
        w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        _, yout, xout, _ = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]
    else:
        w, cin, y, x = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        _, _, yout, xout = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]

    depthwise = cin == groups and cin == cout

    # Disallow depthwise path when training, needs BW ops implementation
    depthwise = depthwise and not dc.is_training_enabled() and not is_convtranspose2d
    
    # Disallow depthwise if we are force disabling it via env var
    depthwise = depthwise and ("PYBUDA_DISABLE_DEPTHWISE_CONV2D_DECOMP" not in os.environ or os.environ["PYBUDA_DISABLE_DEPTHWISE_CONV2D_DECOMP"] != "1")

    if channel_last:
        activations = dc.op("reshape", [activations], (w, 1, y * x, cin), output_df=activations.output_df)
    else:
        activations = dc.op("reshape", [activations], (w, 1, cin, y * x), output_df=activations.output_df)
        activations = dc.op(TransposeTM.create(2, 3), [activations], output_df=activations.output_df)

    weights = transform_weights_for_conv2d(dc, weights, cin, cout, depthwise, groups, kH, kW)

    if is_convtranspose2d and stride_transpose > 1:
        # Create a tensor that dilates input when stride_transpose > 1
        transpose_tensor = create_conv2d_transpose_input_act_dident(y, x, stride_transpose, tile_align=True)
        y *= stride_transpose
        x *= stride_transpose
        yout = yout_transpose
        xout = xout_transpose

    result = activations
    result = dc.op("pad_tile", [result], (-1, result.shape[3]), output_df=result.output_df)
    result = dc.op("pad_tile", [result], (-2, result.shape[2]), output_df=result.output_df)

    padding_same = (padding == [(kW // 2), (kW // 2), (kH // 2), (kH // 2)])
    pad_for_factorization = False
    manual_splice_decomp_th = os.environ.get('PYBUDA_MANUAL_SPLICE_DECOMP_TH')
    sparse_r_padding = ast.literal_eval(os.environ.get('PYBUDA_PAD_SPARSE_MM', "{}"))
    sparse_weight_padding_mm = ast.literal_eval(os.environ.get('PYBUDA_PAD_SPARSE_MM_WEIGHT_MM', "{}"))
    sparse_weight_padding_concat = ast.literal_eval(os.environ.get('PYBUDA_PAD_SPARSE_MM_WEIGHT_CONCAT', "{}"))
    if kH * kW > 1 or (stride[0] > 1 or stride[1] > 1) or not padding_same:
        sparse_r = align_up_tile(yout * xout) // 32
        sparse_c = align_up_tile(result.shape[-2]) // 32
        dense_c = align_up_tile(result.shape[-1]) // 32
        padded_r = 0
        padded_c = 0
        if sparse_r in sparse_r_padding:
            pad_for_factorization = True
            padded_r = sparse_r_padding[sparse_r] - sparse_r

        if dense_c in sparse_weight_padding_mm or dense_c in sparse_weight_padding_concat:
            if dense_c in sparse_weight_padding_concat:
                padded_dense_c = sparse_weight_padding_concat[dense_c]
                pad_dense_c = padded_dense_c - dense_c
                result = dc.op("pad", [result], (0, pad_dense_c*32, 0, False), output_df=result.output_df)
            else: # efficientnet-lite lite1 variant
                padded_dense_c = sparse_weight_padding_mm[dense_c]
                index = torch.arange(result.shape[-1]).tolist()
                spm = torch.sparse_coo_tensor( [index, index], torch.ones(len(index)), (result.shape[-1], padded_dense_c*32), dtype=torch.float32,).unsqueeze(0).unsqueeze(0)
                spm = dc.tensor(spm.to_dense())
                result = dc.op("matmul", [result, spm])

            # pad weight
            padded_weight_len = weights.shape[-2] // dense_c * padded_dense_c
            rows = []
            [rows.extend(torch.arange(offset, offset+dense_c*32).tolist()) for offset in range(0, padded_weight_len, padded_dense_c*32)]
            cols = torch.arange(weights.shape[-2]).tolist()
            spm_weight = torch.sparse_coo_tensor( [rows, cols], torch.ones(len(cols)), (padded_weight_len, weights.shape[-2]), dtype=torch.float32,).unsqueeze(0).unsqueeze(0)
            spm_weight = dc.tensor(spm_weight)
            weights = dc.op("sparse_matmul", [spm_weight, weights])

        if sparse_c in sparse_r_padding:
            pad_for_factorization = True
            padded_c = sparse_r_padding[sparse_c] - sparse_c
            # temporarily add decompotion that manually insert vslice/vstack around splice op
            if manual_splice_decomp_th is not None:
                if sparse_r_padding[sparse_c] >= int(manual_splice_decomp_th):
                    sparse_c_zeros = dc.tensor(torch.zeros((result.shape[-4], padded_c, 32, result.shape[-1])))
                    result = dc.op("vslice", [result], (sparse_c,))
                    result = dc.op("concatenate", [result, sparse_c_zeros], (-3,))
                    result = dc.op("vstack", [result], (sparse_r_padding[sparse_c],))
                else:
                    result = dc.op("pad", [result], (0, 0, 0, padded_c * 32, 0, False))
            else:
                result = dc.op("pad", [result], (0, 0, 0, padded_c * 32, 0, False))

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                # pickers are created row-major, starting from top-left kernel pixel
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(y, x, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True, sparse_r_pad=padded_r, sparse_c_pad=padded_c)
                if is_convtranspose2d and stride_transpose > 1:
                    picker = torch.sparse.mm(picker, transpose_tensor)
                pickers.append(picker)
        sparse = dc.tensor(torch.stack(pickers).unsqueeze(0))
        result = dc.op("sparse_matmul", [sparse, result], output_df=result.output_df)

    if kH * kW > 1:
        result = dc.op("hstack", [result], (kH * kW,), output_df=result.output_df)

    if depthwise:
        result = dc.op("depthwise", [result, weights], (kH * kW,))
    else:
        result = dc.op("matmul", [result, weights])

    if pad_for_factorization:
        # temporarily add decompotion that manually insert vslice/vstack around splice op
        if manual_splice_decomp_th is not None:
            if sparse_r >= int(manual_splice_decomp_th) and sparse_r in sparse_r_padding:
                result = dc.op("vslice", [result], (sparse_r_padding[sparse_r],))
                result = dc.op("select", [result], (-3, 0, sparse_r, sparse_r_padding[sparse_r]))
                result = dc.op("vstack", [result], (sparse_r,))
            else:
                result = dc.op("select", [result], (-2, 0, sparse_r * 32, sparse.shape[-2]))
        else:
            result = dc.op("select", [result], (-2, 0, sparse_r * 32, sparse.shape[-2]))

    result = dc.op("narrow", [result], (3, 0, cout, result.shape[3]), output_df=result.output_df)
    result = dc.op("narrow", [result], (2, 0, yout * xout, result.shape[2]), output_df=result.output_df)

    if bias is not None:
        bias_y, bias_x = 1, 1
        if len(bias.shape) >= 3:
            bias_y, bias_x = bias.shape.as_list()[-2:]
        assert (bias_y == 1 or bias_y == yout) and (bias_x == 1 or bias_x == xout), "bias shape not supported"
        bias = dc.op("reshape", [bias], (1, 1, cout, bias_x * bias_y), output_df=bias.output_df)
        bias = dc.op(TransposeTM.create(-2, -1), [bias], output_df=bias.output_df)
        result = dc.op("add", [result, bias], output_df=bias.output_df)

    if channel_last:
        result = dc.op("reshape", [result], (w, yout, xout, cout))
        dc.fuse(result)
    else:
        result = dc.op(TransposeTM.create(2, 3), [result])
        result = dc.op("reshape", [result], (w, cout, yout, xout))
        dc.fuse(result)


def decompose_conv2d_sparse_second(attr, dc, inputs):
    stride = [attr[0], attr[1],]
    dilation = attr[2]
    groups = attr[3]
    padding = [attr[4], attr[5], attr[6], attr[7],]
    channel_last = attr[-1]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    cout, _, kH, kW = (weights.shape.w, weights.shape.z, weights.shape.r, weights.shape.c)
    if channel_last:
        _, yout, xout, _ = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]
        w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
    else:
        w, cin, y, x = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        _, _, yout, xout = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]

    grouped_conv = groups > 1

    if channel_last:
        activations = dc.op("reshape", [activations], (w, 1, y * x, cin))
    else:
        activations = dc.op("reshape", [activations], (w, 1, cin, y * x))
        activations = dc.op(TransposeTM.create(2, 3), [activations])

    weights = dc.op("reshape", [weights], (1, cout, cin // groups, kH * kW))
    weights = dc.op(TransposeTM.create(1, 3, z_dim_slice=kH*kW), [weights])

    if grouped_conv:
        weights = dc.op("conv2d_grouped_weights", [weights], (groups, kH*kW, cin // groups, cout, True))
    else:
        weights = dc.op("pad_tile", [weights], (-1, cout))
        weights = dc.op(TransposeTM.create(1, 2, z_dim_slice=cin // groups), [weights])
        weights = dc.op("reshape", [weights], (1, 1, cin // groups, align_up_tile(cout) * kH * kW))

    result = activations
    result = dc.op("pad_tile", [result], (-1, cin))
    weights = dc.op("pad_tile", [weights], (-2, weights.shape[2]))

    result = dc.op("matmul", [result, weights])
    result = dc.op("pad_tile", [result], (-2, result.shape[-2]))

    # Always vslice maximally
    v_slice_factor = result.shape[-2] // TILE_DIM
    if v_slice_factor > 1:
        result = dc.op("vslice", [result], (v_slice_factor,))
        result = dc.op(Buffer.create(), [result])  # HW workaround for: tenstorrent/budabackend#656

    if (kH * kW) > 1:
        result = dc.op("hslice", [result], (kH * kW,))
        if "PYBUDA_MIDDLE_CNN_BUFFER" in os.environ: # most workloads are ok without it, and perf is much better... so enable only where needed
            result = dc.op(Buffer.create(), [result])  # HW workaround for: tenstorrent/budabackend#656
        result = dc.op("vstack", [result], (kH * kW,))
        result = dc.op(Buffer.create(), [result])  # HW workaround for: tenstorrent/budabackend#656

    if v_slice_factor > 1:
        result = dc.op("vstack", [result], (v_slice_factor,))

    padding_same = (padding == [(kW // 2), (kW // 2), (kH // 2), (kH // 2)])
    if kH * kW > 1 or (stride[0] > 1 or stride[1] > 1) or not padding_same:
        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(
                    y,
                    x,
                    y_shift,
                    x_shift,
                    kH,
                    kW,
                    stride,
                    padding,
                    dilation,
                    tile_align=True)
                pickers.append(picker)

        # Split the sparse tensor
        sparse = interleave_tiles(pickers)  # to match max vslice after conv matmul
        # sparse = torch.stack(vslice(sparse, self.sparse_mm_t), dim=0).unsqueeze(0)
        sparse_tensor = dc.tensor(sparse)
        result = dc.op("sparse_matmul", [sparse_tensor, result])

    result = dc.op("narrow", [result], (3, 0, cout, result.shape[3]))
    result = dc.op("narrow", [result], (2, 0, yout * xout, result.shape[2]))

    if bias is not None:
        result = dc.op("add", [result, bias])

    if channel_last:
        result = dc.op("reshape", [result], (w, yout, xout, cout))
        dc.fuse(result)
    else:
        result = dc.op(TransposeTM.create(2, 3), [result])
        result = dc.op("reshape", [result], (w, cout, yout, xout))
        dc.fuse(result)


def decompose_conv3d_sparse_first(attr, dc, inputs):
    stride = [attr[0], attr[1], attr[2],]
    dilation = attr[3]
    groups = attr[4]
    padding = [attr[5], attr[6], attr[7], attr[8], attr[9], attr[10]]
    channel_last = attr[-1]

    activations = inputs[0]
    weights = inputs[1]
    bias = inputs[2] if len(inputs) == 3 else None

    #### for now, assuming input (inC, inD, inH, inW), no batch support ####
    cout, _, kD, kH, kW = (weights.shape.v, weights.shape.w, weights.shape.z, weights.shape.r, weights.shape.c)
    #if channel_last:
    #    _, yout, xout, _ = conv2d_out_shape('conv2d', attr, [activations.shape, weights.shape])[0]
    #    w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
    #else:
    w, cin, din, y, x = (activations.shape.v, activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
    _, _, dout, yout, xout = conv3d_out_shape('conv3d', attr, [activations.shape, weights.shape])[0]

    def create_conv2d_sparse_matrix(
        y,
        x,
        kH,
        kW,
        stride,
        padding,
        dilation,
        tile_align=True,
    ):
        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                # pickers are created row-major, starting from top-left kernel pixel
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(y, x, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=tile_align, sparse_r_pad=0, sparse_c_pad=0)
                pickers.append(picker)
        return torch.stack(pickers)

    def vstack(x, factor):
        shape = x.shape
        assert shape[-3] % factor == 0, f"VStack requires Z to be divisible by slice size"
        return x.reshape(shape[:-3] + (shape[-3] // factor, shape[-2] * factor, shape[-1]))

    #grouped_conv = groups > 1
    #depthwise = cin == groups and cin == cout

    #depthwise_env_enabled = "PYBUDA_ENABLE_DEPTHWISE" in os.environ and os.environ["PYBUDA_ENABLE_DEPTHWISE"] == "1"
    # Disallow depthwise path when training, needs BW ops implementation
    #depthwise = depthwise and depthwise_env_enabled and not dc.is_training_enabled()

    # -------------- activations ----------- #
    #if channel_last:
    #    activations = dc.op("reshape", [activations], (w, 1, y * x, cin))
    #else:
    activations = dc.op("reshape", [activations], (w, 1, cin*din, y*x))
    activations = dc.op(TransposeTM.create(-2, -1), [activations])
 
    result = activations
    result = dc.op("pad_tile", [result], (-1, result.shape[3]))
    result = dc.op("pad_tile", [result], (-2, result.shape[2]))

    padding_same = (padding == [(kW // 2), (kW // 2), (kH // 2), (kH // 2)]) 
    if kH * kW > 1 or (stride[0] > 1 or stride[1] > 1) or not padding_same:
        sparse_pt = create_conv2d_sparse_matrix(y, x, kH, kW, stride[0:2], padding[0:4], dilation=dilation,)
        sparse = dc.tensor(sparse_pt.unsqueeze(0))
        result = dc.op("sparse_matmul", [sparse, result])

    if kH * kW > 1:
        result = dc.op("hstack", [result], (kH * kW,))

    # ------------ weights -------------- #
    weights = dc.op("reshape", [weights], (cout, cin, kD, kH * kW))
    weights = dc.op(TransposeTM.create(1, 3, kH * kW), [weights])
    weights = dc.op(TransposeTM.create(0, 1), [weights])

    pickers = []
    for _ in range(kH*kW):
        pickers_inter = []
        for _ in range(cout):
            spm = create_conv2d_sparse_matrix(1, din, 1, kD, (1, stride[2]), [padding[4], padding[5], 0, 0], dilation=dilation, tile_align=False)
            pickers_inter.append(spm)
        spm2 = torch.stack(pickers_inter)
        pickers.append(spm2)
    weights_sparse = torch.stack(pickers) # (kH*kW, cout, kD, dout, din)
    weights_sparse = weights_sparse.to_dense()
    weights_sparse = weights_sparse.permute(0, 1, 3, 4, 2) # (kH*kW, cout, dout, din, kD)
    weights_sparse = vstack(weights_sparse, dout) # (kH*kW, cout, 1, dout*din, kD)
    weights_sparse = torch.squeeze(weights_sparse, -3) # (kH*kW, cout, dout*din, kD)
    weights_sparse = weights_sparse.to_sparse()
    weights_sparse = dc.tensor(weights_sparse)
    weights = dc.op("sparse_matmul", [weights_sparse, weights]) # (kH * kW, outC, outD * iD, inC)
    weights = dc.op("reshape", [weights], (kH*kW, cout, dout, din, cin))
    weights = dc.op(TransposeTM.create(-1, -2), [weights])
    weights = dc.op("reshape", [weights], (1, kH*kW, cout*dout, din*cin))
    weights = dc.op(TransposeTM.create(-1, -2), [weights])
    weights = dc.op("pad_tile", [weights], (-2, din*cin))
    weights = dc.op("pad_tile", [weights], (-1, cout*dout))
    weights = dc.op("vstack", [weights], (kH*kW,))

    # ------------- matmul --------- #
    #if depthwise:
    #    result = dc.op("depthwise", [result, weights], (kH * kW,))
    #else:
    result = dc.op("matmul", [result, weights])

    # ---------- post-processing -------- #
    result = dc.op("narrow", [result], (3, 0, cout * dout, result.shape[3]))
    result = dc.op("narrow", [result], (2, 0, yout * xout, result.shape[2]))

    if bias is not None:
        result = dc.op("add", [result, bias])

    #if channel_last:
    #    result = dc.op("reshape", [result], (w, yout, xout, cout))
    #    dc.fuse(result)
    #else:
    result = dc.op(TransposeTM.create(2, 3), [result])
    result = dc.op("reshape", [result], (cout, dout, yout, xout))
    dc.fuse(result)


def eval(type, attr, ops):
    assert len(ops) <= 3, "Conv ops should have up to three inputs"
    t_ops = to_torch_operands(*ops)

    activations = t_ops[0]
    weights = t_ops[1]
    bias = t_ops[2] if len(t_ops) == 3 else None

    if type == "conv2d":
        assert len(attr) == 9 or len(attr) == 13, f"Conv2d invalid num attributes: {len(attr)}"
        stride = [attr[0], attr[1],]
        dilation = attr[2]
        groups = attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7],]

        channel_last = attr[-1]
        if channel_last:
            activations = activations.permute((0,3,1,2))

        padded_activations = torch.nn.functional.pad(
            activations, 
            padding,
        )
        if (t_ops[1].dtype == torch.int8):
            target_dtype = torch.int32
            padded_activations, weights = padded_activations.float(), weights.float()
            if bias is not None:
                bias = bias.float()
        else:
            target_dtype = torch.float32
            
        result = torch.nn.functional.conv2d(
            padded_activations,
            weights,
            bias=bias,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )
        
        if channel_last:
            result = result.permute((0,2,3,1))

        result = result.to(target_dtype)
        return result
    
    elif type == "conv2d_transpose":
        # Take only 1 padding
        assert len(attr) == 9 or len(attr) == 13, f"Conv2d invalid num attributes: {len(attr)}"
        stride = [attr[0], attr[1],]
        dilation = attr[2]
        groups = attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7],]
        assert all([p == padding[0] for p in padding]), "ConvTranspose2d only supports same padding on all sides"

        channel_last = attr[-1]
        if channel_last:
            activations = activations.permute((0,3,1,2))

        # Building a ConvTranspose2d object so we can use _output_padding().
        # Just passing the activations forward through these detaches the weights
        # from the network, and thus grad will not be generated for it on the .backwards() calls
        tconv = torch.nn.ConvTranspose2d(
            activations.shape[1],
            weights.shape[1] * groups,
            kernel_size=weights.shape[2],
            stride=stride,
            padding=padding[0],
            dilation=dilation,
            groups=groups,
            bias=True if len(t_ops) == 3 else False,
        )

        output_padding = tconv._output_padding(
            activations, None, stride, padding[0], weights.shape[2], dilation)

        result = torch.nn.functional.conv_transpose2d(
            activations, weights, bias, stride, padding[0],
            output_padding, groups, dilation)
        if channel_last:
            result = result.permute((0,2,3,1))
        return result

    elif type == "conv3d":
        assert len(attr) == 12, f"Conv3d invalid num attributes: {len(attr)}"
        stride = [attr[0], attr[1], attr[2],]
        dilation = attr[3]
        groups = attr[4]
        padding = [attr[5], attr[6], attr[7], attr[8], attr[9], attr[10],]

        channel_last = attr[-1]
        if channel_last:
            activations = activations.permute((0,4,1,2,3))

        padded_activations = torch.nn.functional.pad(
            activations, 
            padding,
        )
        result = torch.nn.functional.conv3d(
            padded_activations,
            weights,
            bias=bias,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )
        
        if channel_last:
            result = result.permute((0,2,3,4,1))

        return result

def shape(type, attr, ops):
    if type == "conv3d":
        return conv3d_out_shape(type, attr, ops)
    else:
        return conv2d_out_shape(type, attr, ops)


def lower(type, attr, lc, ops, outputs):
    assert False, "Convolution lowering is intentionally unimplemented"


def backward(type, attr, ac, operand, inputs, output, grad):
    assert False, "Convolution backward is intentionally unimplemented"


def decompose(type, attr, dc, inputs):
    if type == "conv2d":
        assert len(inputs) == 2 or len(inputs) == 3, "Conv2d should have two or three inputs"
        assert len(attr) == 13, f"Conv2d invalid num attributes: {len(attr)}"

        # Decompose gets called recursively on all the newly added nodes as well, so the first pass here might
        # call the decompose_conv2d_prestride(), and then the second one could call decompose_conv2d_sparse_first()

        # We allow prestriding only in inference mode currently
        compiler_cfg = dc.get_compiler_cfg()
        is_prestride_enabled = not dc.is_training_enabled() and compiler_cfg.enable_conv_prestride

        if is_prestride_enabled and should_prestride(attr, dc, inputs):
            # Prestride
            decompose_conv2d_prestride(attr, dc, inputs)
        elif should_fracture_conv_at_op_level(attr, dc, inputs):
            # Fracture
            decompose_fracture_conv2d_at_op_level(attr, dc, inputs)
        elif bool(int(os.environ.get("PYBUDA_CONV2D_SPARSE_SECOND", "0"))):
            # Sparse second
            decompose_conv2d_sparse_second(attr, dc, inputs)
        else:
            # Sparse first (default)
            decompose_conv2d_sparse_first(attr, dc, inputs)

    elif type == "conv2d_transpose":
        assert len(inputs) == 2 or len(inputs) == 3, "Conv2d should have two or three inputs"
        assert len(attr) == 9, f"Conv2d invalid num attributes: {len(attr)}"

        stride = [attr[0], attr[1],]
        dilation = attr[2]
        groups = attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7],]
        assert all([p == padding[0] for p in padding]), "ConvTranspose2d only supports same padding on all sides"
        assert all([s == stride[0] for s in stride]), "ConvTranspose2d only supports same strides"
        stride = stride[0]
        channel_last = attr[-1]

        assert dilation == 1, "Only support dilation == 1 for now"
        activations = inputs[0]
        weights = inputs[1]
        bias = inputs[2] if len(inputs) == 3 else None

        if channel_last:
            _, yout, xout, _ = conv2d_out_shape('conv2d_transpose', attr, [activations.shape, weights.shape])[0]
            w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        else:
            w, cin, y, x = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
            _, _, yout, xout = conv2d_out_shape('conv2d_transpose', attr, [activations.shape, weights.shape])[0]

        _, cout, kH, kW = (weights.shape.w, weights.shape.z * groups, weights.shape.r, weights.shape.c)

        # Transform padding from convtranspose2d space to conv2d
        actual_padding = [
            dilation * (kW - 1) - padding[0],
            dilation * (kW - 1) - padding[1],
            dilation * (kH - 1) - padding[2],
            dilation * (kH - 1) - padding[3],
        ]
        depthwise = (cin == groups) and (cout == cin)

        # stride > 1 means we dilate the input activations
        if stride > 1:
            actual_padding = [dilation * (kH - 1) - x - i % 2 for i, x in enumerate(padding)]

            actual_padding = [
                dilation * (kW - 1) - padding[0],
                dilation * (kW - 1) - padding[1] - 1,
                dilation * (kH - 1) - padding[2],
                dilation * (kH - 1) - padding[3] - 1,
            ]

        # convtranspose2d requires matrix rotation
        weights = rotate_convtranspose2d_weights(dc, weights, cin, cout, depthwise, groups, kH, kW)

        inps = [activations, weights]
        if bias != None:
            inps.append(bias)

        is_convtranspose2d = True
        result = dc.op("conv2d", inps, [1, 1, dilation, groups] + actual_padding + [is_convtranspose2d, yout, xout, stride, channel_last])

        dc.fuse(result)

    elif type == "conv3d": 
        decompose_conv3d_sparse_first(attr, dc, inputs)

def initial_flops_estimate(type, attr, ops):
    macc = 0
    if type == "conv2d":
        output_shape = conv2d_out_shape(type, attr, ops)[0]
        channel_last = attr[-1]
        groups = attr[3]
        if channel_last:
            h_out, w_out = output_shape[-3], output_shape[-2]
        else:
            h_out, w_out = output_shape[-2], output_shape[-1]
        cout, cin, kernel_h, kernel_w = ops[1]
        macc = kernel_h * kernel_w * h_out * w_out * cin * cout / groups

        assert macc > 0

    elif type == "conv2d_transpose":
        output_shape = conv2d_out_shape(type, attr, ops)[0]
        channel_last = attr[-1]
        if channel_last:
            h_out, w_out = output_shape[-3], output_shape[-2]
        else:
            h_out, w_out = output_shape[-2], output_shape[-1]

        cin, cout, kernel_h, kernel_w = ops[1]
        stride_h, stride_w = attr[0], attr[1]

        macc = cin * cout * h_out * w_out * (kernel_h / stride_h) * (kernel_w / stride_w)
        assert macc > 0
    flops = int(macc) * 2
    return flops
