# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import struct

from typing import List, Tuple, Union
from math import prod

import torch
import tensorflow as tf
import numpy as np

from collections import defaultdict
from loguru import logger

from pybuda._C.backend_api import OpModelDesc
from pybuda._C.balancer import FusedSubOpModel, OpModel

from ...pybudaglobal import TILE_DIM

from ...tensor import narrow_buda_tensor_to_pytorch, pad_pytorch_tensor_to_buda, buda_dataformat_to_pytorch_dtype
from pybuda import DataFormat, MathFidelity

def to_torch_operands(*ops):
    """
    Convert input tensors into torch tensors.
    """

    for o in ops:
        assert isinstance(o, (int, torch.Tensor)), f"Invalid operand type: {type(o)}"

    if len(ops) == 2:
        if ops[0].is_floating_point() and ops[1].is_floating_point() and ops[0].dtype != ops[1].dtype:
            ops = (ops[0], ops[1].type(ops[0].dtype))

    if len(ops) == 3:
        ops = list(ops)
        if ops[0].is_floating_point() and ops[1].is_floating_point() and ops[0].dtype != ops[1].dtype:
            ops[1] = ops[1].type(ops[0].dtype)

        if ops[0].is_floating_point() and ops[2].is_floating_point() and ops[0].dtype != ops[2].dtype:
            ops[2] = ops[2].type(ops[0].dtype)

        ops = tuple(ops)

    return ops

def cast_for_cpu_eval(t_ops, op_name=None):
    # Torch does not support int8 or float16 on CPU
    # So we cast to float32
    # Note that INT8 matmul results in INT32 output
    original_type = t_ops[0].dtype
    t_ops = list(t_ops)
    for index, op in enumerate(t_ops):
        if op.dtype == torch.float16:
            t_ops[index] = op.to(torch.float32)
        if op.dtype == torch.int8:
            t_ops[index] = op.to(torch.float32)
            if op_name == "matmul" or op_name == "depthwise":
                original_type = torch.int32
            elif op_name == "sparse_matmul":
                original_type = torch.int8
    return t_ops, original_type

def create_constant_tensor_from_value(value: float, dims: Tuple[int, int], is_buda: bool, df: DataFormat) -> torch.Tensor:
    dim_r, dim_c = dims
    if dim_r < 0:
        dim_r = TILE_DIM if is_buda else 0
    if dim_c < 0:
        dim_c = TILE_DIM if is_buda else 0

    tensor_r = dim_r
    tensor_c = dim_c
    if is_buda and (tensor_c % 32 != 0):
        tensor_c += 32 - tensor_c % 32
    if is_buda and (tensor_r % 32 != 0):
        tensor_r += 32 - tensor_r % 32

    dtype = buda_dataformat_to_pytorch_dtype(df)
    if is_buda:
        tensor = torch.zeros(1, 1, tensor_r, tensor_c, dtype=dtype)
        tensor[:, :, 0:dim_r, 0:dim_c] = value
    else:
        if tensor_c == 0 and tensor_r == 0:
            tensor = torch.tensor(value, dtype=dtype)
        elif tensor_c == 0 or tensor_r == 0:
            dim = tensor_c if tensor_r == 0 else tensor_r
            tensor = torch.zeros(dim, dtype=dtype)
            tensor[0:dim] = value
        else:
            tensor = torch.zeros(tensor_r, tensor_c, dtype=dtype)
            tensor[0:tensor_r, 0:tensor_c] = value

    return tensor

def create_constant_tensor_from_tile(tile: List[float], is_buda: bool, df: DataFormat) -> torch.Tensor:

    assert is_buda, "Tile tensors should only be created for buda graphs"
    assert len(tile) == TILE_DIM * TILE_DIM, "Incorrect number of elements in tile"
    tensor = torch.FloatTensor(tile)
    tensor = tensor.reshape(1, 1, TILE_DIM, TILE_DIM)
    tensor = tensor.type(buda_dataformat_to_pytorch_dtype(df))
    return tensor

def create_constant_tensor_from_tensor(tensor_values: List[float], tensor_shape: List[int], is_buda: bool, df: DataFormat) -> torch.Tensor:
    assert prod(tensor_shape) == len(tensor_values)
    tensor = torch.FloatTensor(tensor_values)
    tensor = tensor.reshape(tensor_shape)
    if is_buda:
        tensor = pad_pytorch_tensor_to_buda(tensor, [])
    tensor = tensor.type(buda_dataformat_to_pytorch_dtype(df))
    return tensor

def create_constant_tensor(flat_data: List[float], shape: List[int], is_buda: bool, df: DataFormat) -> torch.Tensor:
    tensor = torch.FloatTensor(flat_data)
    tensor = tensor.reshape(*shape)
    if is_buda:
        tensor = pad_pytorch_tensor_to_buda(tensor)
    tensor = tensor.type(buda_dataformat_to_pytorch_dtype(df))
    return tensor


def dump_tensor(tensor, name, entry=0):
    fmt = 'f'
    if tensor.dtype == torch.half or tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float)
    elif tensor.dtype == torch.int or tensor.dtype == torch.int32:
        fmt = 'i'
    elif tensor.dtype == torch.short or tensor.dtype == torch.int16:
        fmt = 'h'
    elif tensor.dtype == torch.int8:
        fmt = 'b'
    elif tensor.dtype == torch.uint8:
        fmt = 'B'

    with open(f"{name}.{entry}.bin", 'wb') as out_file:
        for val in tensor.ravel().tolist():
            out_file.write(struct.pack(fmt, val)) 
        out_file.close()

def eval_debug_print(type, inputs, output):
    mode = int(os.environ.get("EVAL_DEBUG", "0"))
    if mode > 0:
        print(f"{type}: inputs: {[i.shape if isinstance(i, torch.Tensor) else 1 for i in inputs]}, output: {output.shape}")
        if mode > 1:
            for i, input in enumerate(inputs):
                print("input ", i)
                print(input)
            print("output:")
            print(output)

def calculate_pcc(a, b):
    if torch.all(torch.isnan(a)) and torch.all(torch.isnan(b)):
        logger.warning("Both tensors are 'nan'")
        return 1.0 

    if torch.all(torch.isnan(a)) or torch.all(torch.isnan(b)):
        logger.error("One tensor is all nan, the other is not.")
        return 0.0

    # Test if either is completely zero
    if torch.any(a.bool()) != torch.any(b.bool()):
        return 0.0

    #if torch.any(torch.isinf(a)) or torch.any(torch.isinf(b)):
    #    raise RuntimeError(f"Tensor overflow to infinity: \n{a}\n{b}")

    #if torch.any(torch.isneginf(a)) or torch.any(torch.isneginf(b)):
    #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{a}\n{b}")

    # For now, mask all infs and nans so that we check the rest... TODO
    a = a.clone()
    a[torch.logical_or(torch.isnan(a), torch.logical_or(torch.isinf(a), torch.isneginf(a)))] = 0
    b = b.clone()
    b[torch.logical_or(torch.isnan(b), torch.logical_or(torch.isinf(b), torch.isneginf(b)))] = 0

    if torch.equal(a, b):
        return 1.0

    if a.dtype == torch.bfloat16:
        a = a.type(torch.float32)
        b = b.type(torch.float32)
    pcc = np.min(
            np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(a).detach().numpy()).flatten(), 
                np.ma.masked_invalid(torch.squeeze(b).detach().numpy()).flatten()
        ))

    if isinstance(pcc, np.ma.core.MaskedConstant):
        return 1.0

    return pcc

def compare_tensor_to_golden(name: str, golden: Union[torch.Tensor, tf.Tensor, tf.Variable], calculated: torch.Tensor, is_buda=False, rtol=None, atol=None, pcc=None, warning_only=False, relative_atol = None, verify_cfg = None):
    # Convert golden to pytorch tensor for comparisons
    if isinstance(golden, (tf.Tensor, tf.Variable)):
        golden = torch.from_numpy(golden.numpy())

    if golden.dtype == torch.bool and calculated.dtype != torch.bool:
        calculated = calculated.to(torch.bool)

    if golden.dtype == torch.bool and calculated.dtype == torch.bool:
        return bool(torch.all(golden == calculated))

    if is_buda:
        calculated = narrow_buda_tensor_to_pytorch(calculated, golden.shape)

    if rtol is None or (isinstance(rtol, dict) and (golden.dtype not in rtol or rtol[golden.dtype] is None)):
        if verify_cfg is not None and golden.dtype in verify_cfg.rtol and verify_cfg.rtol[golden.dtype] is not None:
            rtol = verify_cfg.rtol[golden.dtype]
        else:
            rtol = 0 # use atol only, rtol is unreliable for very small numbers
    elif isinstance(rtol, dict):
        rtol = rtol[golden.dtype]

    if atol is None or (isinstance(atol, dict) and (golden.dtype not in atol or atol[golden.dtype] is None)):
        if verify_cfg is not None and golden.dtype in verify_cfg.atol and verify_cfg.atol[golden.dtype] is not None:
            atol = verify_cfg.atol[golden.dtype]
        else:
            if relative_atol is None and verify_cfg is not None:
                relative_atol = verify_cfg.relative_atol
            if relative_atol is None:
                relative_atol = 0.1

            if torch.all(torch.isnan(golden)):
                atol = 0
            else:
                max_value = (torch.max(torch.abs(golden[~torch.isnan(golden)]))).item()
                atol = max_value * relative_atol # allow up to 'relative_atol' error
    elif isinstance(atol, dict):
        atol = atol[golden.dtype]

    if pcc is None and verify_cfg is not None:
        pcc = verify_cfg.pcc

    while len(calculated.shape) > len(golden.shape) and calculated.shape[0] == 1:
        calculated = calculated.squeeze(0)

    while len(golden.shape) > len(calculated.shape) and golden.shape[0] == 1:
        golden = golden.squeeze(0)

    if not golden.shape == calculated.shape:
        logger.error("Tensor shape mismatch on {}", name)
        logger.debug("Golden: (shape = {}", golden.shape)
        logger.debug("Calculated: (shape = {}", calculated.shape)
        return False

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    ok = torch.allclose(golden, calculated, rtol=rtol, atol=atol, equal_nan=True)
    callback_ok = True if verify_cfg is None or verify_cfg.golden_compare_callback is None else verify_cfg.golden_compare_callback(golden, calculated)
    ok &= callback_ok
    pcc_value = 0
    if not (pcc is None or golden.flatten().size() == (1,)): # PCC for single values doesn't work
        pcc_value = calculate_pcc(golden, calculated)
        if pcc_value >= pcc and not ok:
            logger.warning("PCC is correct but allclose failed on {}", name)
            logger.trace("Golden: (shape = {}", golden.shape)
            logger.trace(golden)
            logger.trace("Calculated: (shape = {}", calculated.shape)
            logger.trace(calculated)
            logger.warning("Max ATOL Delta: " + "{:.3e}".format(torch.max(torch.abs(golden - calculated)).item()) + ", atol=" +  "{}".format(atol))
            logger.warning("Max RTOL Delta: " + "{:.3e}".format(torch.max(torch.abs(golden - calculated)/calculated).item()) + ", rtol=" + "{}".format(rtol) )
        ok |= pcc_value >= pcc

    if not ok:
        if warning_only:
            logger.warning("Tensor mismatch on {}", name)
        else:
            logger.error("Tensor mismatch on {}", name)
        logger.trace("Golden: (shape = {}", golden.shape)
        logger.trace(golden)
        logger.trace("Calculated: (shape = {}", calculated.shape)
        logger.trace(calculated)
        logger.info("Max ATOL Delta: " + "{:.3e}".format(torch.max(torch.abs(golden - calculated)).item()) + ", atol=" +  "{}".format(atol))
        logger.info("Max RTOL Delta: " + "{:.3e}".format(torch.max(torch.abs(golden - calculated)/calculated).item()) + ", rtol=" + "{}".format(rtol) )
        if pcc is not None:
            logger.info("PCC got={}, required={}", pcc_value, pcc)
        if not callback_ok:
            logger.info("User golden_compare_callback returned False")
        #torch.set_printoptions(profile="full")
        #print(golden-calculated)
        #torch.set_printoptions(profile="default")
        if not warning_only:
            return False
    else:
        if os.environ.get("SHOW_MATCHING", "0") != "0":
            logger.trace("Golden: (shape = {}", golden.shape)
            logger.trace(golden)
            logger.trace("Calculated (correct): (shape = {}", calculated.shape)
            logger.trace(calculated)
            if pcc is not None:
                logger.debug("PCC (correct) got={}, required={}", pcc_value, pcc)
        logger.debug("Tensors match on {}", name)

    return True

def math_fidelity_to_multiplier(fidelity: MathFidelity) -> int:
    if fidelity == MathFidelity.LoFi:
        return 1
    if fidelity == MathFidelity.HiFi2:
        return 2
    if fidelity == MathFidelity.HiFi3:
        return 3
    return 4


def data_format_to_int(df: DataFormat) -> int:
    if df == DataFormat.Float16:
        return 1
    if df == DataFormat.Float16_b:
        return 2
    if df == DataFormat.Bfp8:
        return 3
    if df == DataFormat.Bfp8_b:
        return 4
    if df == DataFormat.Bfp4:
        return 5
    if df == DataFormat.Bfp4_b:
        return 6
    if df == DataFormat.Bfp2:
        return 7
    if df == DataFormat.Bfp2_b:
        return 8
    if df == DataFormat.Float32:
        return 9
    if df == DataFormat.Lf8:
        return 11
    raise RuntimeError(f"Unknown data format {df}")

def op_model_to_desc(type: str, arch_name: str, op_model: OpModel, sub_op_model: FusedSubOpModel = None) -> OpModelDesc:
    desc = OpModelDesc()
    desc.arch = arch_name
    desc.data_format = op_model.data_format
    desc.math_fidelity = op_model.math_fidelity()
    desc.t = op_model.output_buffers[0].block_shape.t
    desc.approx_mode = False

    if op_model.op_type() == "fused_op":
        desc.type = sub_op_model.type
        desc.mblock_m = sub_op_model.mblock_m
        desc.mblock_n = sub_op_model.mblock_n
        desc.ublock_rt = sub_op_model.ublock_rt
        desc.ublock_ct = sub_op_model.ublock_ct

        if (desc.type == "matmul"):
            desc.mblock_k = sub_op_model.mblock_k
            desc.ublock_kt = sub_op_model.ublock_kt
        elif (desc.type == "reduce"):
            desc.op_attr = sub_op_model.reduce_dim

        desc.approx_mode = "PYBUDA_EXP_APPROX" in os.environ
    else:
        desc.type = type
        desc.mblock_m = op_model.output_buffers[0].block_shape.mblock_m
        desc.mblock_n = op_model.output_buffers[0].block_shape.mblock_n
        desc.ublock_rt = op_model.output_buffers[0].block_shape.ublock.rt
        desc.ublock_ct = op_model.output_buffers[0].block_shape.ublock.ct

        if type == "matmul":
            if op_model.is_sparse_matmul:
                desc.ublock_kt = op_model.input_buffers[1].block_shape.ublock.rt
                desc.mblock_k = op_model.op_shape.inputs[1].rt // desc.ublock_kt
                desc.sparse_indices = op_model.sparse_indices
                if os.environ.get("PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES", False):
                    desc.sparse_nz_ublocks = op_model.nz_ublocks
                    desc.sparse_nz_strips = op_model.nz_strips

                    # op model descriptor assumes grid_size [1, 1], so we need to scale down the number of
                    # sparse tiles, ublocks and strips to what is expected to end up on a single core
                    if os.environ.get("PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS", False):
                        if op_model.nz_tiles > 1:
                            desc.sparse_indices = max(op_model.nz_tiles // op_model.grid_shape.r, 1)
                        else:
                            desc.sparse_indices = op_model.nz_tiles

                        if op_model.nz_ublocks > 1:
                            desc.sparse_nz_ublocks = max(op_model.nz_ublocks // op_model.grid_shape.r, 1)

                        if op_model.nz_strips > 1:
                            desc.sparse_nz_strips = max(op_model.nz_strips // op_model.grid_shape.r, 1) 
                else:
                    # old sparse estimates
                    if os.environ.get("PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS", False):
                        if op_model.sparse_indices > 1:
                            desc.sparse_indices = max(op_model.sparse_indices // op_model.grid_shape.r, 1)
            else:
                desc.ublock_kt = op_model.input_buffers[0].block_shape.ublock.ct
                desc.mblock_k = op_model.op_shape.inputs[0].ct // desc.ublock_kt

                # requant/dequant part of matmul is calculated separately for now, and we need to pass
                # matmul output format here
                if "requant" in op_model.buda_op_attrs() or "dequant" in op_model.buda_op_attrs():
                    desc.data_format = DataFormat.Int32

        if type == "depthwise":
            desc.mblock_k = op_model.op_shape.inputs[1].rt
            desc.ublock_kt = 1

        desc.op_attr = op_model.get_reduce_dim()
        # desc.op_attr is only used to capture the dim of reduce - ideally, we should support tt::BudaOpAttrs in
        # tt_op_model_desc - when we do, uncomment the line below
        # desc.op_attr = op_model.buda_op_attrs()

        # If reduce_z, we manually copy the "z" param to special field in tt_op_model_desc - we should pass all buda attrs
        if type == "reduce" and op_model.buda_op_attrs()["dim"] == "z":
            desc.reduce_z = op_model.buda_op_attrs()["z"]

    attrs = op_model.buda_op_attrs()
    # If the attributes contain approximate mode set it.
    if 'approximate_mode' in attrs:
        desc.approx_mode = attrs['approximate_mode'] == 'true'

    return desc

def calculate_tile_size(val):
    # We might not even care about large dim size 
    # that are not divisible by 32
    if (val > 32):
        return 32

    smallest_pad = 31
    current_tile_size = 32

    tile_sizes = [32, 16, 8, 4, 2, 1]

    for tile_size_ in tile_sizes:
        rem = val % tile_size_
        pad = tile_size_ - rem
        if (rem == 0 and smallest_pad != 0):
            # Pick the largest tile size that divides evenly
            smallest_pad = 0
            current_tile_size = tile_size_
        elif (pad <= smallest_pad):
            # pick the tile size with smallest pad
            smallest_pad = pad
            current_tile_size = tile_size_


    return current_tile_size

# Global compiler cache
g_compiler_perf_cache : defaultdict = defaultdict(dict)

def get_compiler_cached_cycles(desc: OpModelDesc) -> int:
    global g_compiler_perf_cache

    if not g_compiler_perf_cache:
        cache_file = os.environ.get("PYBUDA_COMPILER_CACHE", None)
        if cache_file is not None and os.path.exists(cache_file):
            with open(os.environ["PYBUDA_COMPILER_CACHE"], 'rb') as file:
                import pickle
                g_compiler_perf_cache = pickle.load(file)
        else:
            return None

    cached_op_model = g_compiler_perf_cache["op_model"]

    if desc.type in cached_op_model:
        cache_cycles = cached_op_model[desc.type]
        shapes = (desc.mblock_m, desc.mblock_n, desc.ublock_rt, desc.ublock_ct, desc.t)

        if desc.type == 'matmul':  # append k dim to lookup
            shapes = shapes + (desc.mblock_k, desc.ublock_kt)

        if shapes in cache_cycles:
            cycle_count = cache_cycles[shapes]
            # print(f"Using recorded cycle count for {desc.type} of shapes {shapes} -> {cycle_count}")
            return cycle_count

    return None
