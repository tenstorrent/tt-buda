# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from typing import OrderedDict
from pybuda.pybudaglobal import TILE_DIM
from pybuda.tvm_to_python import compile_tvm_to_python
from pybuda.verify.config import TestKind
import pytest
from sqlalchemy import true
from test.tvm.python.test_sanity import test_linear

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from loguru import logger
import pybuda
from pybuda import (
    Tensor,
    PyTorchModule,
    PyBudaModule,
    VerifyConfig,
)
from pybuda.config import CompilerConfig, _get_global_compiler_config, CompileDepth
from pybuda.verify.backend import verify_module

class AlexnetReshape(torch.nn.Module):
    def __init__(self, newshape, oldshape):
        super().__init__()
        self.newshape = newshape
        self.a = torch.randn(oldshape).transpose(-2, -1)
        self.t = torch.randn(newshape).transpose(-2, -1)

    def forward(self, activation):
        subtracted = activation - 2
        subtracted = torch.softmax(subtracted, -1)
        reshaped = torch.reshape(subtracted, self.newshape)
        summed = reshaped + 2
        return torch.softmax(summed, -1)

def align_up_tile(num, TILE_DIM):
    if num % TILE_DIM == 0:
        return num
    
    return ((num // TILE_DIM) + 1) * TILE_DIM

def create_padding_x_shift_sparse_picker_matrix(padded_length, data_per_slice, pad_per_slice, num_data_rows, stack_factor):
   

    cols = []
    lo = 0
    hi = data_per_slice
    while lo < num_data_rows:
        cols.extend(torch.arange(lo, hi).tolist())
        lo += data_per_slice + pad_per_slice
        hi += data_per_slice + pad_per_slice
    rows = []
    lo = 0
    for _ in range(stack_factor):
        rows.extend(torch.arange(lo, lo + len(cols)//stack_factor).tolist())
        lo = align_up_tile(len(cols)//stack_factor, TILE_DIM)

    return torch.sparse_coo_tensor(
        [rows, cols],
        torch.ones(len(cols)),
        (align_up_tile(len(rows), TILE_DIM), padded_length),
        dtype=torch.float32
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

def create_reshape_flatten_sparse_picker_matrix(orig_r, new_r, tile_dim=TILE_DIM):
    cols = torch.arange(new_r//tile_dim)
    rows = cols * tile_dim
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (new_r, orig_r),
        dtype=torch.float32,
    )

def create_flattened_padding_removal_sparse_picker_matrix(r, start, stop, length):
    num_pads = r // length
    cols = []
    [cols.extend((torch.arange(start, stop) + (length * pad)).tolist()) for pad in range(num_pads)]
    cols = torch.tensor(cols)
    rows = torch.arange(num_pads * stop)
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (num_pads * stop, r),
        dtype=torch.float32,
    )

def create_padding_insert_sparse_picker(length, padded_length, slices):
    rows = []
    slice_size = length // slices
    padded_slice_size = align_up_tile(slice_size, TILE_DIM)
    [rows.extend(torch.arange(s*padded_slice_size, s*padded_slice_size + slice_size).tolist()) for s in range(slices)]
    cols = torch.arange(length).tolist()
    return torch.sparse_coo_tensor(
        [rows, cols],
        torch.ones(len(cols)),
        (slices*padded_slice_size, padded_length),
        dtype=torch.float32,
    )

def get_factors(x):
    facs = []
    for i in range(1, x + 1):
        if x % i == 0:
            facs.append(i)
    return facs

input_shapes = [[1, 95, 17, 99], [1, 50, 55, 35], [1, 133, 81, 81], [1, 65, 98, 60], [1, 20, 78, 21]]

@pytest.mark.parametrize("input_shape", input_shapes)
def test_reshape(input_shape):
    pos_slices = get_factors(input_shape[-1])[1:-1]

    slc = pos_slices[0]
    newshape = input_shape[:2] + [input_shape[2]*slc] + [input_shape[3]//slc]
    torch.set_printoptions(threshold=100000, linewidth=100000)

    print(f"SHAPE: {input_shape} -> {newshape}", end=" ")
    # slice_factor = newshape[-1] // input_shape[-1]
    x = torch.rand(input_shape)
    goal = torch.reshape(x, newshape)
    goalt = torch.transpose(goal, -2, -1)

    in1 = pybuda.Tensor.create_from_torch(x)
    padx = pybuda.op.PadTile("", in1, -1, input_shape[-2])
    pady = pybuda.op.PadTile("", padx, -2, input_shape[-1])
    # padded = [0, 0] + [(pady.shape[2]-1) * TILE_DIM] + [0]
    # pady = pybuda.op.Pad("", pady, padded)

    spm = create_reshape_flatten_sparse_picker_matrix(pady.shape[-2], TILE_DIM*pady.shape[-2])
    spm = pybuda.Tensor.create_from_torch(spm)
    mm = pybuda.op.SparseMatmul("", spm, pady)
    result = pybuda.op.VSlice("", mm, pady.shape[-2])

    if input_shape[-1] % TILE_DIM == 0:
        hstk = pybuda.op.HSlice("", result, slc)
        result = pybuda.op.VStack("", hstk, hstk.shape[-3] // input_shape[-3])
    else:
        spm = create_padding_insert_sparse_picker(input_shape[-1], pady.shape[-1], slc)
        spm = pybuda.Tensor.create_from_torch(spm)
        t = pybuda.op.Transpose("", result, -2, -1)
        mm = pybuda.op.SparseMatmul("", spm, t)
        vslc = pybuda.op.VSlice("", mm, slc)
        t = pybuda.op.Transpose("", vslc, -2, -1)
        result = pybuda.op.VStack("", t, t.shape[-3] // input_shape[-3])

    spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, 1, TILE_DIM)
    spm = torch.nn.functional.pad(spm.to_dense(), (0, 0, 0, align_up_tile(spm.shape[-2], TILE_DIM) - spm.shape[-2]), mode='constant', value=0).to_sparse()
    spm = pybuda.Tensor.create_from_torch(spm)
    result = pybuda.op.SparseMatmul("", spm, result)
    if not align_up_tile(newshape[-2], TILE_DIM) == align_up_tile(result.shape[-2], TILE_DIM):
        spm = create_flattened_padding_removal_sparse_picker_matrix(result.shape[-2], 0, newshape[-2], newshape[-2])
        spm = pybuda.Tensor.create_from_torch(spm)
        result = pybuda.op.SparseMatmul("", spm, result)

    n1 = pybuda.op.Narrow("", result, -1, 0, newshape[-1], result.shape[-1])
    res = pybuda.op.Narrow("", n1, -2, 0, newshape[-2], n1.shape[-2])

    assert torch.all(goal == res.value())

    mod = PyTorchModule("axelnet_reshape", AlexnetReshape(newshape, input_shape))
    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER # Avoid no grids error
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(),
    )


def pad_to_tile_dim(n, TILE_DIM):
    if n % TILE_DIM == 0:
        return n
    return n + TILE_DIM - (n % TILE_DIM)

def test_xy_flatten():
    TILE_DIM = 32
    input_shape = (1, 32, 112, 112)
    newshape = (1, 1, input_shape[-3], input_shape[-2]*input_shape[-1])

    in1 = pybuda.Tensor.create_from_torch(torch.rand(input_shape))
    padx = pybuda.op.PadTile("", in1, -1, input_shape[-1])
    pady = pybuda.op.PadTile("", padx, -2, input_shape[-2])
    # pady = pybuda.op.Pad("", in1, (0, TILE_DIM-input_shape[-2], 0, TILE_DIM-input_shape[-1]))
    torch.set_printoptions(threshold=100000, linewidth=10000)

    if input_shape[-3] > 1:
        pady = pybuda.op.HStack("", pady, input_shape[-3])

    padded_shape = pady.shape
    r_new = padded_shape[-1] * input_shape[-2] // (padded_shape[-1] // TILE_DIM)
    fl_spm = pybuda.op.eval.create_reshape_flatten_sparse_picker_matrix(pady.shape[-2], r_new, TILE_DIM)
    fl_spm = pybuda.Tensor.create_from_torch(fl_spm)
    mm = pybuda.op.SparseMatmul("", fl_spm, pady)

    if input_shape[-3] > 1:
        mm = pybuda.op.HSlice("", mm, input_shape[-3])
        
    rt = align_up_tile(r_new, TILE_DIM) // TILE_DIM
    vs = pybuda.op.VSlice("", mm, rt)
    hs = pybuda.op.HStack("", vs, rt)

    if input_shape[-3] > 1:
        hs = pybuda.op.VStack("", hs, input_shape[-3])

    if input_shape[-1] % TILE_DIM:
        tx = pybuda.op.Transpose("", hs, -2, -1)
        pr_spm = pybuda.op.eval.create_flattened_padding_removal_sparse_picker_matrix(hs.shape[-1], 0, input_shape[-1], pad_to_tile_dim(input_shape[-1], TILE_DIM))

        pr_spm = pybuda.Tensor.create_from_torch(pr_spm)
        pr_mm = pybuda.op.SparseMatmul("", pr_spm, tx)
        utx = pybuda.op.Transpose("", pr_mm, -2, -1)
    else:
        utx = hs

    if input_shape[-3] > 1:
        pr_spm = pybuda.op.eval.create_flattened_padding_removal_sparse_picker_matrix(utx.shape[-2], 0, 1, TILE_DIM)
        pr_spm = pybuda.Tensor.create_from_torch(pr_spm)
        res = pybuda.op.SparseMatmul("", pr_spm, utx)
    else:
        res = pybuda.op.Narrow("", utx, -2, 0, 1, utx.shape[-2])

    assert torch.all(in1.value().reshape(newshape) == res.value())

    # mod = PyTorchModule("alexnet_reshape", AlexnetReshape(newshape))

    # _get_global_compiler_config().enable_consteval = False
    # verify_module(
    #     mod,
    #     (input_shape,),
    #     verify_cfg=VerifyConfig(),
    # )

# input_shapes = [(1, 1, 6, 6)]#[(1, int(torch.randint(1, 513, (1,))), int(torch.randint(1, 513, (1,))), int(torch.randint(1, 513, (1,)))) for i in range(1)]
# newshapes = [(1, 6, 3, 2)]

# @pytest.mark.parametrize("newshape", newshapes)
# @pytest.mark.parametrize("input_shape", input_shapes)
# def test_y_slice(input_shape, newshape):

#     def factors(n):
#         factors = []
#         for i in range(1, n+1):
#             if n%i == 0:
#                 factors.append(i)

#         return factors
#     TILE_DIM=8
#     np.random.seed(int(1000*time.time()) % (2**32 -1))
#     y_factors = factors(input_shape[-1])[1:-1]
#     y_slice = y_factors[np.random.randint(len(y_factors))]
#     # while y_slice == 1: y_slice = y_factors[np.random.randint(len(y_factors))]
    
#     y_slice = newshape[-2]

#     # newshape = list(input_shape[:-3]) + [input_shape[-2]] + [y_slice] + [input_shape[-1] // y_slice]
#     print(f"{input_shape} --> {newshape}")

#     in1 = pybuda.Tensor.create_from_torch(torch.rand(input_shape))
#     padx = pybuda.op.PadTile("", in1, -1, input_shape[-1])
#     pady = pybuda.op.PadTile("", padx, -2, input_shape[-2])
#     torch.set_printoptions(threshold=100000, linewidth=10000)
#     num_rows_per_slice = input_shape[-1] // y_slice
#     padded_y_slice = pady.shape[-1] // num_rows_per_slice
#     vslc = pybuda.op.VSlice("", pady, pady.shape[-2])

#     fl_spm = pybuda.op.eval.create_reshape_flatten_sparse_picker_matrix(1, vslc.shape[-1])
#     fl_spm = pybuda.Tensor.create_from_torch(fl_spm)
#     mm = pybuda.op.SparseMatmul("", fl_spm, mm)


def test_zx_transpose():
    TILE_DIM = 32
    input_shape = (1, 1280, 1, 1)
    newshape = (1, 1, 1, 1280)

    in1 = pybuda.Tensor.create_from_torch(torch.rand(input_shape))
    padx = pybuda.op.PadTile("", in1, -1, input_shape[-1])
    pady = pybuda.op.PadTile("", padx, -2, input_shape[-2])

    vs = pybuda.op.VStack("", pady, 1280)
    fspm = pybuda.op.eval.create_reshape_flatten_sparse_picker_matrix(1280, 40960).transpose(-1, -2)
    fspm = pybuda.Tensor.create_from_torch(fspm)
    mm = pybuda.op.SparseMatmul("", fspm, vs)
    tx = pybuda.op.Transpose("", mm, -2, -1)
    
    res = pybuda.op.Narrow("", tx, -2, 0, 1, 32)
    assert torch.all(in1.value().reshape(newshape) == res.value())
    mod = PyTorchModule("axelnet_reshape", AlexnetReshape(newshape, input_shape))

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(),
    )

class TestConcat(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, in1, in2):
        in1 = in1 + 2
        in2 = in2 + 4
        reshaped = torch.cat((in1, in2), dim=self.axis)
        summed = reshaped + 2
        return summed

def align_up_tile(num, TILE_DIM=TILE_DIM):
    if num % TILE_DIM == 0:
        return num
    
    return ((num // TILE_DIM) + 1) * TILE_DIM

def test_concat():
    pytest.skip()
    TILE_DIM = 32
    input_shape_1 = (1, 128, 16, 64)
    input_shape_2 = (1, 128, 16, 32)
    axis = -1

    in1 = pybuda.Tensor.create_from_torch(torch.rand(input_shape_1))
    in2 = pybuda.Tensor.create_from_torch(torch.rand(input_shape_2))
    concat = pybuda.op.Concatenate("", in1, in2, axis=axis)

    if TILE_DIM == 32:
        padx1 = pybuda.op.PadTile("", in1, -1, input_shape_1[-1])
        pady1 = pybuda.op.PadTile("", padx1, -2, input_shape_1[-2])
        padx2 = pybuda.op.PadTile("", in2, -1, input_shape_2[-1])
        pady2 = pybuda.op.PadTile("", padx2, -2, input_shape_2[-2])
    else:
        pad1 = [0, 0, 0, 0]
        if input_shape_1[-1] % TILE_DIM:
            pad1[-3] = TILE_DIM - (input_shape_1[-1] % TILE_DIM)
        if input_shape_1[-2] % TILE_DIM:
            pad1[-1] = TILE_DIM - (input_shape_1[-2] % TILE_DIM)
        pady1 = pybuda.op.Pad("", in1, pad1)

        pad2 = [0, 0, 0, 0]
        if input_shape_2[-1] % TILE_DIM:
            pad2[-3] = TILE_DIM - (input_shape_2[-1] % TILE_DIM)
        if input_shape_2[-2] % TILE_DIM:
            pad2[-1] = TILE_DIM - (input_shape_2[-2] % TILE_DIM)
        pady2 = pybuda.op.Pad("", in2, pad2)
    
    if axis == -1:
        pady1 = pybuda.op.Transpose("", pady1, -2, -1)

    if axis == -1:
        pady2 = pybuda.op.Transpose("", pady2, -2, -1)

    vs1 = pady1
    vs2 = pady2
    if axis == -3:
        vs1 = stacker("", pady1, input_shape_1[-3])
    vs2 = pady2
    if axis == -3:
        vs2 = stacker("", pady2, input_shape_2[-3])

    if axis >= -2:
        # r_new = align_up_tile((input_shape_1[axis] + input_shape_2[axis]) * input_shape_1[-3], TILE_DIM)
        # mm1_len = input_shape_1[axis] * input_shape_1[-3]
        r_new = align_up_tile((input_shape_1[axis] + input_shape_2[axis]) * 1, TILE_DIM)
        mm1_len = input_shape_1[axis] * 1
        start, length = 0, TILE_DIM
        num_pads = align_up_tile(mm1_len, TILE_DIM)//TILE_DIM
        stop = TILE_DIM if input_shape_1[axis] % TILE_DIM == 0 else input_shape_1[axis]
        cols = []
        [cols.extend((torch.arange(start, stop) + (length * pad)).tolist()) for pad in range(num_pads)]
        cols = torch.tensor(cols)
        rows = torch.arange(len(cols))
    else:
        cols = torch.arange(vs1.shape[-2])
        rows = torch.arange(vs1.shape[-2])
        r_new = vs1.shape[-2] + vs2.shape[-2]
    spm1 = torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (r_new, vs1.shape[-2]),
        dtype=torch.float32,
    )
    spm1 = pybuda.Tensor.create_from_torch(spm1)
    mm1 = pybuda.op.SparseMatmul("", spm1, vs1)

    if axis >= -2:
        # mm2_len = input_shape_2[axis] * input_shape_1[-3]
        mm2_len = input_shape_2[axis] * 1
        num_pads = align_up_tile(mm2_len, TILE_DIM)//TILE_DIM
        stop = TILE_DIM if input_shape_2[axis] % TILE_DIM == 0 else input_shape_2[axis]
        cols = []
        [cols.extend((torch.arange(start, stop) + (length * pad)).tolist()) for pad in range(num_pads)]
        cols = torch.tensor(cols)
        rows = torch.arange(len(cols)) + mm1_len
    else:
        cols = torch.arange(vs1.shape[-2])
        rows = torch.arange(vs1.shape[-2])
        r_new = vs1.shape[-2] + vs2.shape[-2]
    spm2 = torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (r_new, vs2.shape[-2]),
        dtype=torch.float32,
    )
    spm2 = pybuda.Tensor.create_from_torch(spm2)
    mm2 = pybuda.op.SparseMatmul("", spm2, vs2)

    add = pybuda.op.Add("", mm1, mm2)
    vsl = add
    if axis == -3:
        rt = r_new//pady1.shape[-2]
        vsl = pybuda.op.HSlice("", add, rt)
    if axis == -1:
        vsl = pybuda.op.Transpose("", vsl, -2, -1)

    narrow_shape = input_shape_1[-1] if axis != -1 else input_shape_1[-1] + input_shape_2[-1]
    if narrow_shape != vsl.shape[-1]:
        vsl = pybuda.op.Narrow("", vsl, -1, 0, narrow_shape, vsl.shape[-1])
    narrow_shape = input_shape_1[-2] if axis != -2 else input_shape_1[-2] + input_shape_2[-2]
    if narrow_shape != vsl.shape[-2]:
        vsl = pybuda.op.Narrow("", vsl, -2, 0, narrow_shape, vsl.shape[-2])

    assert torch.all(vsl.value() == concat.value())
    mod = PyTorchModule("concat_decomp", TestConcat(axis))

    verify_module(
        mod,
        (input_shape_1, input_shape_2,),
        verify_cfg=VerifyConfig(),
    )


input_shapes = [[1, 1, 240, 137], [1, 1, 75, 38], [1, 1, 180, 184], [1, 1, 25, 222], [1, 1, 24, 86]]

@pytest.mark.parametrize("input_shape", input_shapes)
def test_vslice(input_shape):
    # TILE_DIM = 32
    # input_shape = [1, 64, 231, 160] #if aligned else (1, 1, 256, 36)
    # newshape = [1, 2, 6, 8]

    pos_slices = get_factors(input_shape[-2])[1:-1]

    slc = pos_slices[0]
    
    orig_shape = input_shape
    newshape = [1, slc, input_shape[-2]//slc, input_shape[-1]]
    attr = newshape

    x = torch.randn(input_shape)
    goal = x.reshape(newshape)
    result = pybuda.Tensor.create_from_torch(x)
    result = pybuda.op.PadTile("", result, -1, result.shape[-1])
    result = pybuda.op.PadTile("", result, -2, result.shape[-2])

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
    slice_factor = attr[-3]
    if attr[-2] >= TILE_DIM:
        spm1 = create_flattened_padding_removal_sparse_picker_matrix(spm.shape[-2], 0, slice_factor*padded_dim, spm.shape[-2])
        spm = torch.sparse.mm(spm1, spm)

    spm = pybuda.Tensor.create_from_torch(spm)
    result = pybuda.op.SparseMatmul("", spm, result)
    result = pybuda.op.VSlice("", result, slice_factor)

    assert align_up_tile(result.shape[-2]) == align_up_tile(attr[-2])
    result = pybuda.op.Narrow("", result, -1, 0, attr[-1], result.shape[-1])
    result = pybuda.op.Narrow("", result, -2, 0, attr[-2], result.shape[-2])

    assert torch.all(goal == result.value())

    mod = PyTorchModule("axelnet_reshape", AlexnetReshape(newshape, input_shape))
    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER # Avoid no grids error
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(),
    )


input_shapes = [[1, 177, 184, 79], [1, 176, 186, 119], [1, 172, 4, 59], [1, 104, 199, 11], [1, 145, 3, 132]]

@pytest.mark.parametrize("input_shape", input_shapes)
def test_vstack(input_shape):
    
    orig_shape = input_shape
    newshape = [1, 1, input_shape[-2]*input_shape[-3], input_shape[-1]]# newshape = [1, 2, 7, 2]
    # newshape = [1, 1, 14, 2]
    attr = newshape
    print(f"SHAPE: {input_shape} -> {newshape}", end=" ")

    x = torch.randn(input_shape)
    goal = x.reshape(newshape)
    result = pybuda.Tensor.create_from_torch(x)
    result = pybuda.op.PadTile("", result, -1, result.shape[-1])
    result = pybuda.op.PadTile("", result, -2, result.shape[-2])
    padded_shape = result.shape
    slice_factor = orig_shape[-3]
    result = pybuda.op.VStack("", result, slice_factor)

    if orig_shape[-2] % TILE_DIM != 0:
        # Pick out multiple rows in a tile
        num_rows = (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
        num_slices = orig_shape[-3]
        rows = torch.arange(attr[-2]).tolist()
        cols = []

        for i in range(num_slices):
            cols.extend((torch.arange(orig_shape[-2]) + (i * padded_shape[-2])).tolist())

        s_pick_multi_row = torch.sparse_coo_tensor(
            [rows, cols],
            torch.ones(len(cols)),
            (num_rows, result.shape[-2]),
            dtype=torch.float32,
        )
        s_pick_multi_row = pybuda.Tensor.create_from_torch(s_pick_multi_row)
        result = pybuda.op.SparseMatmul("", s_pick_multi_row, result)

    assert align_up_tile(result.shape[-2]) == align_up_tile(attr[-2])
    result = pybuda.op.Narrow("", result, -1, 0, attr[-1], result.shape[-1])
    result = pybuda.op.Narrow("", result, -2, 0, attr[-2], result.shape[-2])

    assert torch.all(goal == result.value())

    mod = PyTorchModule("axelnet_reshape", AlexnetReshape(newshape, input_shape))
    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER # Avoid no grids error
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(),
    )
    
    
input_shapes = [[1, 59, 48, 266], [1, 49, 2, 238], [1, 22, 32, 158], [1, 45, 28, 219], [1, 9, 32, 257]]

@pytest.mark.parametrize("input_shape", input_shapes)
def test_full_flatten(input_shape): 
    # input_shape = [1, 86, 192, 193]
    size = 1
    for d in input_shape:
        size *= d
    
    print(f"SHAPE: {input_shape}")
    newshape = [1, 1, 1, size]
    
    x = torch.randn(input_shape)
    goal = torch.reshape(x, newshape)
    in1 = pybuda.Tensor.create_from_torch(x)
    padx = pybuda.op.PadTile("", in1, -1, input_shape[-1])
    pady = pybuda.op.PadTile("", padx, -2, input_shape[-2])
    result = pady

    if len(input_shape) > 2 and input_shape[-3] != 1:
        result = pybuda.op.VStack("", pady, input_shape[-3])
        if input_shape[-2] % TILE_DIM:
            spm = create_padding_shift_sparse_picker_matrix(input_shape[-3]*input_shape[-2], input_shape[-3], result.shape[-2])
            spm = pybuda.Tensor.create_from_torch(spm)
            mm = pybuda.op.SparseMatmul("", spm, result)
            result = mm
    
    padded_shape = result.shape
    r_new = TILE_DIM * padded_shape[-2]
    spm = create_reshape_flatten_sparse_picker_matrix(padded_shape[-2], r_new)
    spm = pybuda.Tensor.create_from_torch(spm)
    result = pybuda.op.SparseMatmul("", spm, result)
    rt = align_up_tile(r_new) // TILE_DIM
    
    vslc = pybuda.op.VSlice("", result, rt)
    hstk = pybuda.op.HStack("", vslc, rt)
    result = hstk

    t = pybuda.op.Transpose("", hstk, -2, -1)
    spm = create_flattened_padding_removal_sparse_picker_matrix(t.shape[-2], 0, input_shape[-1], padded_shape[-1])
    rows = torch.arange(0, align_up_tile(newshape[-1])).tolist()
    cols = rows
    spm2 = torch.sparse_coo_tensor((rows, cols), torch.ones(len(rows)), (len(rows), spm.shape[-2]), dtype=torch.float32)
    spm = torch.sparse.mm(spm2, spm)
    spm = pybuda.Tensor.create_from_torch(spm)
    mm = pybuda.op.SparseMatmul("", spm, t)
    result = pybuda.op.Transpose("", mm, -2, -1)

    assert align_up_tile(newshape[-1]) == align_up_tile(result.shape[-1])
    assert align_up_tile(newshape[-2]) == align_up_tile(result.shape[-2])
    
    result = pybuda.op.Narrow("", result, -2, 0, newshape[-2], result.shape[-2])
    result = pybuda.op.Narrow("", result, -1, 0, newshape[-1], result.shape[-1])
    
    assert torch.all(result.value() == goal)
    
    mod = PyTorchModule("axelnet_reshape", AlexnetReshape(newshape, input_shape))
    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER # Avoid no grids error
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(),
    )
    
    
indexes = list(range(30, 35))
lengths = list(range(30, 35))

@pytest.mark.parametrize("index", indexes)
@pytest.mark.parametrize("length", lengths)
def test_single_select(test_device, index, length):
    test_kind = TestKind.INFERENCE
    input_shape = (1, 1, 128, 64)
    stride = input_shape[-2]
    class Model(PyBudaModule):
        
        def __init__(self, inp_shape):
            super().__init__("single_elect_test")
            
            x = pybuda.Tensor.create_from_torch(torch.randn((1, 128, 8 ,8)))
            self.y = pybuda.op.tm.Select("", x, -3, (index, length), stride).shape[-3]
            
            self.add_constant("one")
            self.add_constant("two")
            self.add_parameter("three", pybuda.Parameter(torch.randn(128, 128)))
            self.add_parameter("four", pybuda.Parameter(torch.randn(self.y, self.y)))
            
            self.set_constant("one", torch.randn(1, 128, 1, 1))
            self.set_constant("two", torch.randn(1, 128, 1, 1))
            
        
        def forward(self, x):
            # x = (1, 1, 32, 64)
            x = pybuda.op.Matmul("", self.get_parameter("three"), x)
            x = pybuda.op.Reshape("", x, (1, 128, 8, 8))
            x = pybuda.op.Multiply("", x, self.get_constant("one"))
            x = pybuda.op.Add("", x, self.get_constant("two"))
            x = pybuda.op.Select("", x, -3, (index, length), stride)
            x = pybuda.op.Reshape("", x, (1, 1, self.y, 64))
            x = pybuda.op.Matmul("", self.get_parameter("four"), x)
            return x
            
    model = Model(input_shape)
    
    model(pybuda.Tensor.create_from_torch(torch.randn(input_shape)))

    module = model

    
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

def test_multi_select(test_device):
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        pytest.skip("Skip this test for golden Wormhole B0")
    test_kind = TestKind.INFERENCE
    input_shape = (1, 1, 30, 3072)
    class Model(PyBudaModule):
        
        def __init__(self, inp_shape):
            super().__init__("multi_select_test")
            
            self.add_constant("one")
            self.add_constant("two")
            self.set_constant("one", torch.randn(1, 30, 1, 1))
            self.set_constant("two", torch.randn(1, 30, 1, 1))
            
        
        def forward(self, x):
            # x = (1, 1, 32, 64)
            # import pdb; pdb.set_trace()
            x = pybuda.op.Softmax("", x, dim=-1)
            r0 = pybuda.op.Reshape("", x, (1, 30, 4, 768))
            mult0 = pybuda.op.Multiply("", r0, self.get_constant("one"))
            add0 = pybuda.op.Add("", mult0, self.get_constant("two"))
            
            # Multiple selects where the dim we are selecting from is unaffected by commute
            # The reshapes are all inverses to r0, should be erased
            s1 = pybuda.op.Select("", add0, -3, (0, 10), 30)
            s2 = pybuda.op.Select("", add0, -3, (10, 10), 30)
            s3 = pybuda.op.Select("", add0, -3, (20, 10), 30)
            r1 = pybuda.op.Reshape("", s1, (1, 1, 10, 3072))
            r2 = pybuda.op.Reshape("", s2, (1, 1, 10, 3072))
            r3 = pybuda.op.Reshape("", s3, (1, 1, 10, 3072))
            m1 = pybuda.op.Softmax("", r1, dim=-1)
            m2 = pybuda.op.Softmax("", r2, dim=-1)
            m3 = pybuda.op.Softmax("", r3, dim=-1)
            
            # Multiple selects where the dim we are selecting from is unaffected by commute
            # The reshapes are NOT inverses to r0. An inverse reshape shall be placed on top of each one
            # and subsequently erased during the optimize pass
            s4 = pybuda.op.Select("", add0, -3, (0, 10), 30)
            s5 = pybuda.op.Select("", add0, -3, (10, 10), 30)
            s6 = pybuda.op.Select("", add0, -3, (20, 10), 30)
            r4 = pybuda.op.Reshape("", s4, (1, 10, 16, 192))
            r5 = pybuda.op.Reshape("", s5, (1, 10, 16, 192))
            r6 = pybuda.op.Reshape("", s6, (1, 10, 16, 192))
            m4 = pybuda.op.Softmax("", r4, dim=-1)
            m5 = pybuda.op.Softmax("", r5, dim=-1)
            m6 = pybuda.op.Softmax("", r6, dim=-1)
            
            # Multiple selects where the dim we are selecting from IS affected by commute
            # The reshapes are NOT inverses to r0. An inverse reshape shall be placed on top of each one
            # and subsequently erased during the optimize pass
            s7 = pybuda.op.Select("", add0, -1, (0, 256), 768)
            s8 = pybuda.op.Select("", add0, -1, (256, 256), 768)
            s9 = pybuda.op.Select("", add0, -1, (512, 256), 768)
            r7 = pybuda.op.Reshape("", s7, (1, 30, 16, 64))
            r8 = pybuda.op.Reshape("", s8, (1, 30, 16, 64))
            r9 = pybuda.op.Reshape("", s9, (1, 30, 16, 64))
            m7 = pybuda.op.Softmax("", r7, dim=-1)
            m8 = pybuda.op.Softmax("", r8, dim=-1)
            m9 = pybuda.op.Softmax("", r9, dim=-1)
            
            # Multiple selects where the dim we are selecting from IS affected by commute
            # The reshapes are all inverses to r0, should be erased
            s10 = pybuda.op.Select("", add0, -1, (0, 256), 768)
            s11 = pybuda.op.Select("", add0, -1, (256, 256), 768)
            s12 = pybuda.op.Select("", add0, -1, (512, 256), 768)
            r10 = pybuda.op.Reshape("", s10, (1, 1, 30, 1024))
            r11 = pybuda.op.Reshape("", s11, (1, 1, 30, 1024))
            r12 = pybuda.op.Reshape("", s12, (1, 1, 30, 1024))
            m10 = pybuda.op.Softmax("", r10, dim=-1)
            m11 = pybuda.op.Softmax("", r11, dim=-1)
            m12 = pybuda.op.Softmax("", r12, dim=-1)
            
            return m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12
            
    model = Model(input_shape)
    
    model(pybuda.Tensor.create_from_torch(torch.randn(input_shape)))

    module = model

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    

def test_binary_stack_on_x():
    
    class Model(PyBudaModule):
        def forward(self, x, y):
            # import pdb; pdb.set_trace()
            out1 = pybuda.op.BinaryStack("", x, y, -1)
            
            return out1
    
    input_shape = (1, 2, 4, 4)        

    verify_module(
        Model("binary_stack"),
        (input_shape, input_shape),
        verify_cfg=VerifyConfig(),
    )

input_shapes = [list((1, 3, dim, dim)) for dim in range(4, 33)]
scale_factors = list(range(2, 17))
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("scale_factor", scale_factors)
def test_downsample_nearest(input_shape, scale_factor):
    
    if input_shape[-1] % scale_factor != 0:
        pytest.skip("input_shape must be divisible by scale_factor")
    
    class Downsample(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            out = torch.nn.functional.interpolate(x, scale_factor=1/scale_factor, mode='nearest')
            return out
    
    verify_module(
        PyTorchModule("downsample", Downsample()),
        (input_shape,),
        verify_cfg=VerifyConfig(),
    )


input_shapes = [list((1, 3, dim, dim*2)) for dim in range(4, 20)]
scale_factors = list(range(2, 10))
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("scale_factor", scale_factors)
def test_downsample_2d_nearest_channel_first_pytorch(test_device, input_shape, scale_factor):

    if input_shape[-1] % scale_factor != 0 or input_shape[-2] % scale_factor != 0:
        pytest.skip("input_shape must be divisible by scale_factor")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    class downsample_2d_model(torch.nn.Module):
        def __init__(self, scale_factor):
            super().__init__()
            self.scale_factor = scale_factor

        def forward(self, input_tensor):
            return torch.nn.functional.interpolate(input_tensor, scale_factor=1/self.scale_factor, mode='nearest')

    model = downsample_2d_model(scale_factor=scale_factor)
    model.eval()

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(
        "pt_downsample_2d", model
    )

    input_sample = torch.rand(input_shape)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(input_sample.shape,)],
        inputs=[(input_sample,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework=True,
            verify_tvm_compile=True,
        ),
    )


input_shapes = [list((1, dim, dim*2, 3)) for dim in range(4, 20)]
scale_factors = list(range(2, 10))
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("scale_factor", scale_factors)
def test_downsample_2d_nearest_channel_last_pytorch(test_device, input_shape, scale_factor):

    if input_shape[1] % scale_factor != 0 or input_shape[2] % scale_factor != 0:
        pytest.skip("input_shape must be divisible by scale_factor")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    class Downsample2d(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, input):
            return pybuda.op.Resize2d("", input, sizes=[input_shape[1] // scale_factor, input_shape[2] // scale_factor], method="nearest_neighbor", channel_last=True)

    model = Downsample2d("Downsample2d_channel_last")


    verify_module(
            model,
            (input_shape,),
            verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            )
    )


def get_factorization(n):
    factors = []
    i = 2
    while i*i <= n:
        while (n % i) == 0:
            factors.append(i)
            n //= i
        i += 1
    if n > 1:
       factors.append(n)
    return factors

begin_end_shapes = [([1, 47, 36, 1], [1, 3, 2, 282]), ([1, 38, 27, 15], [1, 19, 405, 2]), ([1, 14, 60, 24], [1, 72, 140, 2]), ([1, 54, 1, 22], [1, 18, 33, 2]), ([1, 8, 62, 56], [1, 8, 1736, 2])]
@pytest.mark.parametrize("begin_end_shape", begin_end_shapes)
def test_general_reshape(begin_end_shape):
    class ReshapeModel(PyBudaModule):
        def __init__(self, newshape):
            super().__init__("general_reshape")
            self.add_constant("const")
            self.set_constant("const", torch.eye(newshape[-1]).unsqueeze(0).unsqueeze(0))
            self.newshape = newshape
                    
        def forward(self, a):
            a = pybuda.op.Softmax("", a, dim=-1)
            a = pybuda.op.Reshape("", a, self.newshape)
            a = pybuda.op.Matmul("", a, self.get_constant("const"))
            return a
    
    orig_shape, newshape = begin_end_shape
    logger.debug(f"Testing reshape decomp: {orig_shape} --> {newshape}")
    model = ReshapeModel(newshape)
    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    verify_module(
        model,
        (orig_shape,),
        verify_cfg=VerifyConfig(),
    )

begin_end_shapes = [([1, 1, 224, 1664], [1, 224, 416, 4]), ([1, 1, 32, 1024], [1, 32, 32, 32]), ([1, 1, 40, 1600], [1, 40, 40, 40]), ([1, 1, 20, 200], [1, 20, 100, 2]), ([1, 1, 1280, 4], [1, 1280, 2, 2])]
@pytest.mark.parametrize("begin_end_shape", begin_end_shapes)
def test_xy_unflatten_reshape(begin_end_shape):
    # Tests these kind of reshapes: [1, 1, 32, 1024] -> [1, 32, 32, 32], [1, 1, 1280, 4] -> [1, 1280, 2, 2]
    class ReshapeModel(PyBudaModule):
        def __init__(self, newshape):
            super().__init__("xy_unflatten_reshape")
            self.add_constant("const")
            self.set_constant("const", torch.eye(newshape[-1]).unsqueeze(0).unsqueeze(0))
            self.newshape = newshape
                    
        def forward(self, a):
            a = pybuda.op.Softmax("", a, dim=-1)
            a = pybuda.op.Reshape("", a, self.newshape)
            a = pybuda.op.Matmul("", a, self.get_constant("const"))
            return a
        
    orig_shape, newshape = begin_end_shape
    logger.debug(f"Testing xy unflatten reshape decomp: {orig_shape} --> {newshape}")
    model = ReshapeModel(newshape)
    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    verify_module(
        model,
        (orig_shape,),
        verify_cfg=VerifyConfig(),
    )
    
input_shapes = [[1, 9, 13, 17], [1, 12, 57, 21], [1, 47, 31, 28], [1, 50, 52, 27], [1, 49, 38, 33]]
@pytest.mark.parametrize("shape", input_shapes)
def test_pixel_shuffle(shape):
    r = 2
    shape = shape[:-3] + [r*r*shape[-3]] + shape[-2:]
    
    class PixelShuffleModel(PyBudaModule):
        def __init__(self):
            super().__init__("pixel_shuffle")
                    
        def forward(self, a):
            return pybuda.op.PixelShuffle("", a, r)
    
    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    verify_module(
        PixelShuffleModel(),
        (shape,),
        verify_cfg=VerifyConfig(),
    ) 

def test_reshape_with_smm_padding():
    class ReshapeModelWithPadding(PyBudaModule):
        def __init__(self, name, orig_shape, newshape):
            super().__init__(name)
            dim_width = orig_shape[-2]
            new_dim_width = newshape[-2]
            spm = torch.sparse_coo_tensor([list(range(dim_width)), list(range(dim_width))], torch.ones(dim_width),(dim_width, dim_width),dtype=torch.float32)
            spm = torch.stack([spm] * orig_shape[-3], -3).unsqueeze(0)
            self.add_constant("const")
            self.set_constant("const", pybuda.Tensor.create_from_torch(spm, constant=True))
            spm = torch.sparse_coo_tensor([list(range(new_dim_width)), list(range(new_dim_width))], torch.ones(new_dim_width),(new_dim_width, new_dim_width),dtype=torch.float32)
            spm = torch.stack([spm] * newshape[-3], -3).unsqueeze(0)
            self.add_constant("const2")
            self.set_constant("const2", pybuda.Tensor.create_from_torch(spm, constant=True))
            self.newshape = newshape

        def forward(self, x):
            x = pybuda.op.SparseMatmul("smm0", self.get_constant("const") ,x)
            x = pybuda.op.Reshape("r0", x, self.newshape)
            x = pybuda.op.SparseMatmul("smm1", self.get_constant("const2"), x)
            return x

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{14:16}"

    orig_shape = (1, 256, 14, 14)
    new_shape = (1, 1, 256, 196)
    model = ReshapeModelWithPadding("test_reshape_with_smm_padding", orig_shape, new_shape)
    verify_module(
        model,
        (orig_shape,),
        verify_cfg=VerifyConfig(),
    )

    del os.environ["PYBUDA_PAD_SPARSE_MM"]

