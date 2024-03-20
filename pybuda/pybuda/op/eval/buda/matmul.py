# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentError
from random import random
import numpy as np
import os
import torch

from loguru import logger

import pybuda._C.balancer as balancer
from pybuda._C import DataFormat, MathFidelity
from pybuda._C.backend_api import get_op_model_execution_cycles
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile, align_up
from pybuda._C.graph import UBlockOrder

from ..common import to_torch_operands, cast_for_cpu_eval, math_fidelity_to_multiplier, data_format_to_int, op_model_to_desc, get_compiler_cached_cycles

class CycleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CycleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out) # makes no sense to have negative cycles
        return out

def get_u_kt_bits(u_kt):
    u_kt_bits = 1
    while ((u_kt - 1) >> u_kt_bits) > 0:
        u_kt_bits += 1

    return u_kt_bits


def get_hash_sum_prod(vector, modulo_factor=997):
    hash_sum = 0
    hash_product = 1
    [hash_sum:= (hash_sum + v) % modulo_factor for v in vector]
    [hash_product:= (((hash_product * v) % modulo_factor) * (i // 32 + 1) * (i % 32 + 1)) % modulo_factor for (i, v) in enumerate(vector) if v != 0]

    return hash_sum, hash_product

def strip_ident_matmul(
    sparse_tiles_tensor,
    act,
    encodings,
    sparse_tile_ptr_bits,
    sparse_ublock_idx_bits,
    outer_r,
    outer_d,
    outer_c,
    inner_r,
    inner_d,
    inner_c,
    batch_cnt,
):
    if (sparse_tiles_tensor.dtype == torch.int8 or act.dtype == torch.int8):
        target_dtype = torch.int32
        sparse_tiles_tensor, act = sparse_tiles_tensor.float(), act.float()
    else:
        target_dtype = act.dtype

    rets = []
    ret = torch.zeros((outer_r * inner_r, outer_c * inner_c, TILE_DIM, TILE_DIM))

    u_kt_bits = get_u_kt_bits(u_kt=inner_d)

    print_debug = 0
    print_indices = False

    def to_int16(i):
        as_bytes = list(int.to_bytes(i, byteorder="little", length=4, signed=True))
        return [(as_bytes[1] << 8) | as_bytes[0], (as_bytes[3] << 8) | as_bytes[2]]
    ublock_tile_index_bits = 16 - sparse_tile_ptr_bits
    encodings_tile = -1  # pointer to current encoding tile
    tile_bin = None
    curr_strip_ptr = None
    prev_strip_index = 0

    just_popped_strip_info_tile = True

    while len(rets) < batch_cnt:
        print(f"~~ batch: {len(rets)}") if print_debug else None
        if True:
            if just_popped_strip_info_tile:
                just_popped_strip_info_tile = False
                encodings_tile += 1

                print(f"tile[{encodings_tile}] ") if print_indices else None
                tile_int32 = encodings[0, 0, encodings_tile * TILE_DIM: (encodings_tile + 1) * TILE_DIM, :].reshape(TILE_DIM * TILE_DIM).tolist()
                tile_bin = [i for l in list(map(to_int16, tile_int32)) for i in l]
                curr_strip_ptr = 0

            last_row_tile_strip_index = (tile_bin[curr_strip_ptr + 1] << 16) | tile_bin[curr_strip_ptr]
            last_out = bool(last_row_tile_strip_index & (1 << 30))
            last_strip_in_tile = bool(last_row_tile_strip_index & (1 << 31))
            strip_index = last_row_tile_strip_index & ((1 << 30) - 1)
            assert strip_index >= prev_strip_index, f"Strip index goes backward in t: strip_index[{strip_index}] prev_strip_index[{prev_strip_index}]"
            prev_strip_index = strip_index
            nz_ublocks_in_strip = tile_bin[curr_strip_ptr + 2]
            oid = strip_index
            out_batch = len(rets)
            print(f"  t[{out_batch}] oid[{oid}] strip_ptr[0x{curr_strip_ptr*2:04x}] strip_info_struct{{ .strip_index = {strip_index}, nz_ublocks = {nz_ublocks_in_strip}, .last_strip_in_row = {last_out}, .last_strip_in_tile = {last_strip_in_tile} }}") if print_indices else None
            print(f"~~~~ si: {strip_index}") if print_debug else None
            print(f"~~~~ + last_strip_in_row: {last_out}") if print_debug else None
            print(f"~~~~ + last_strip_in_tile: {last_strip_in_tile}") if print_debug else None

            # Load in1 strip of data here

            if True:
                print(f"~~~~~~ nz_ublocks: {nz_ublocks_in_strip}") if print_debug else None
                current_index = 0
                ublock_start_index = 0
                current_ublock_index = 0
                nz_tiles_in_ublock = 0
                first_tile_index = 0

                ublock_cntr = 0
                out_r = 0
                while out_r < outer_r:
                    ublock_start_index = current_index
                    if ublock_cntr >= nz_ublocks_in_strip:
                        break
                    current_ublock_index = tile_bin[curr_strip_ptr + current_index + 3] & ((1 << sparse_ublock_idx_bits) - 1)
                    left_ublock_zero = (current_ublock_index != out_r)
                    if not left_ublock_zero:
                        ublock_cntr += 1
                    else:
                        out_r = current_ublock_index
                        continue

                    print(f"~~~~~~~~ ublock_index: {current_ublock_index}") if print_debug else None

                    for out_c in range (outer_c):
                        # DEST reload here

                        if not left_ublock_zero:
                            current_index = ublock_start_index
                            encoded = tile_bin[curr_strip_ptr + 3 + current_index]
                            current_index += 1
                            current_ublock_index = encoded & ((1 << sparse_ublock_idx_bits) - 1)
                            nz_tiles_in_ublock = encoded >> sparse_ublock_idx_bits
                            nz_tiles_in_ublock_bits = 16 - sparse_ublock_idx_bits
                            nz_tiles_in_ublock = (1 << nz_tiles_in_ublock_bits) if nz_tiles_in_ublock == 0 else nz_tiles_in_ublock
                            print(f"~~~~~~~~ num_matmuls: {nz_tiles_in_ublock}") if print_debug and out_c == 0 else None
                            first_tile_index = current_index

                            dst_index = 0

                            out_of_tile_range = False
                            for in_r in range (inner_r):
                                for in_d in range(inner_d):
                                    if out_of_tile_range:
                                        break
                                    encoded = tile_bin[curr_strip_ptr + 3 + current_index]
                                    # in1 = encoded & ((1 << ublock_tile_index_bits) - 1)
                                    in1_rt = (encoded & ((1 << ublock_tile_index_bits) - 1)) >> u_kt_bits
                                    in1_ct = (encoded & ((1 << ublock_tile_index_bits) - 1)) & ((1 << u_kt_bits) - 1)
                                    left_tile_zero = (in1_rt != in_r or in1_ct != in_d) or out_of_tile_range
                                    # left_tile_zero = in1 != (in_r * inner_d + in_d) or out_of_tile_range
                                    if not left_tile_zero:
                                        in0_index = encoded >> ublock_tile_index_bits
                                        print(f"~~~~~~~~~~ in0_index: {in0_index}") if print_debug and out_c == 0 else None
                                        print(f"~~~~~~~~~~ in1_index: {in1_rt}, {in1_ct}") if print_debug and out_c == 0 else None
                                        current_index += 1
                                        dst_index = in_r * inner_c
                                        for in_c in range(inner_c):
                                            in1_index = out_c * inner_c * inner_d + in_d * inner_c + in_c

                                            # Matmul here
                                            r_idx_r = strip_index * inner_d + in_d
                                            r_idx_c = out_c * inner_c + in_c
                                            left = sparse_tiles_tensor[0, 0, :, in0_index * TILE_DIM: (in0_index + 1) * TILE_DIM]
                                            right = act[0, 0, r_idx_r * TILE_DIM: (r_idx_r + 1) * TILE_DIM, r_idx_c * TILE_DIM: (r_idx_c + 1) * TILE_DIM]

                                            ret_idx_r = out_r * inner_r + in_r
                                            ret_idx_c = out_c * inner_c + in_c
                                            tile_res = left @ right
                                            ret[ret_idx_r, ret_idx_c] += tile_res

                                            if print_debug:
                                                print(f"~~~~~~~~~~~~ in0[{in0_index}]") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~ act[{oid * inner_d + in_d}, {r_idx_c}]") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~ out[{out_batch}, {ret_idx_r}, {ret_idx_c}]") if print_debug and out_c == 0 else None

                                                in0_sum, in0_prod = get_hash_sum_prod(left.flatten().numpy().view(np.uint32).tolist())
                                                in1_sum, in1_prod = get_hash_sum_prod(right.flatten().numpy().view(np.uint32).tolist())
                                                out_sum, out_prod = get_hash_sum_prod(tile_res.flatten().numpy().view(np.uint32).tolist())

                                                print(f"~~~~~~~~~~~~~~ in0_hash_sum = {in0_sum}") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~~~ in0_hash_prod = {in0_prod}") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~~~ in1_hash_sum = {in1_sum}") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~~~ in1_hash_prod = {in1_prod}") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~~~ out_hash_sum = {out_sum}") if print_debug and out_c == 0 else None
                                                print(f"~~~~~~~~~~~~~~ out_hash_prod = {out_prod}") if print_debug and out_c == 0 else None

                                        if current_index - first_tile_index == nz_tiles_in_ublock:
                                            out_of_tile_range = True

                        # if last_out, pack to out buffer and push, else pack&push to interm buffer
                    out_r += 1

                if last_strip_in_tile:
                    just_popped_strip_info_tile = True

                curr_strip_ptr += 3 + current_index

            if last_out:
                rets.append(ret.transpose(1, 2).reshape(1, 1, outer_r * inner_r * TILE_DIM, outer_c * inner_c * TILE_DIM))
                ret = torch.zeros((outer_r * inner_r, outer_c * inner_c, TILE_DIM, TILE_DIM))

    assert len(rets) == batch_cnt, f"Expected to get {batch_cnt} result(s) from sparse matmul, instead got {len(rets)}"

    rets = [x.to(target_dtype) for x in rets]
    return rets



def eval(type, attr, ops):
    assert len(ops) <= 3, "Matrix multiply should have two or three inputs"
    assert len(attr) <= 15, f"Unexpected number of attrs for matmul: {len(attr)}"

    accumulate = (len(attr) >= 1) and bool(attr[0])
    t_ops = to_torch_operands(*ops)
    is_sparse = len(attr) >= 2 and bool(attr[1])

    op_type = "sparse_matmul" if is_sparse else "matmul"
    t_ops, original_type = cast_for_cpu_eval(t_ops, op_type)

    if type == "matmul" and not is_sparse:
        result = torch.matmul(t_ops[0], t_ops[1])
        result = result.to(original_type)
        if len(t_ops) > 2:
            # broadcast across rows
            result = result + t_ops[2][:, :, 0:1, :]
    elif type == "matmul" and is_sparse:
        assert len(attr) == 15, f"Unexpected number of attrs for sparse matmul: {len(attr)}"
        _, _, sparse_tile_ptr_bits, wdim, zdim, rdim, cdim, fracture_factor, u_rt, u_kt, u_ct, grid_c, t_stream_factor_r, t_stream_factor_c, sparse_ublock_idx_bits = attr
        sparse_tensor, activations, encodings = t_ops

        # Inputs 0 & 2 are tm broadcasted by a factor of (grid_c / fracture_factor)
        # Here we undo the tm broadcast to make the math below easier
        sparse_tensor = sparse_tensor[..., :sparse_tensor.shape[-1] // (grid_c // fracture_factor)]
        encodings = encodings[..., :encodings.shape[-1] // (grid_c // fracture_factor)]

        # If fractured, activations will be broadcast in C by fracture_factor, so we undo it here
        activations = activations.detach()
        activations.requires_grad_(False)
        activations = activations[..., :activations.shape[-1] // fracture_factor]

        inner_r = u_rt
        inner_d = u_kt
        inner_c = u_ct
        outer_r = align_up_tile(rdim) // TILE_DIM // inner_r // fracture_factor
        outer_d = activations.shape[-2] // (TILE_DIM * inner_d)
        outer_c = activations.shape[-1] // TILE_DIM // inner_c
        grid_r = encodings.shape[2] // TILE_DIM

        assert activations.shape[-3] % zdim == 0

        act = activations
        act = act.reshape(1, 1, act.shape[-3] * act.shape[-2], act.shape[-1])

        sparse_width_f = sparse_tensor.shape[-1] // fracture_factor
        encodings_width_f_t = encodings.shape[-1] // fracture_factor // TILE_DIM

        encodings = encodings.reshape(grid_r, fracture_factor * encodings_width_f_t, TILE_DIM, TILE_DIM)

        # TODO: Model this the way it's done in BE golden
        f_results = []
        for f in range(fracture_factor):
            r_results = []
            for g_r in range(grid_r):
                # TODO: Do we want to simulate c dim as well?
                r_sparse_tensor = sparse_tensor[:, :, g_r * TILE_DIM: (g_r + 1) * TILE_DIM, f * sparse_width_f: (f + 1) * sparse_width_f]
                r_encodings = encodings[g_r, f * encodings_width_f_t: (f + 1) * encodings_width_f_t, :, :].view(1, 1, -1, TILE_DIM)
                r_results.append(
                    strip_ident_matmul(
                        r_sparse_tensor,
                        act,
                        r_encodings,
                        sparse_tile_ptr_bits,
                        sparse_ublock_idx_bits,
                        outer_r // grid_r // t_stream_factor_r,
                        outer_d,
                        outer_c,
                        inner_r,
                        inner_d,
                        inner_c,
                        t_stream_factor_r * t_stream_factor_c * zdim,
                    )
                )
            # interleave results
            result = [torch.cat(r_r, dim=-3) for r_r in r_results]
            result = torch.cat(result, dim=-2)
            f_results.append(result)

        result = torch.cat(f_results, dim=-1)
        result = result.to(original_type)

    if accumulate:
        result = torch.sum(result, dim=-3, keepdim=True)

    return result


def shape(type, attr, ops, tile_height, tile_width):
    assert len(ops) in [2, 3, 4], "Matrix multiply should have two or three inputs"

    accumulate = (len(attr) >= 1) and bool(attr[0])
    is_sparse = (len(attr) >= 2) and bool(attr[1])
    if len(ops[1]) > len(ops[0]):
        output_shape = list(ops[1])[:-2] + [ops[0][-2], ops[1][-1]]
    else:
        output_shape = list(ops[0])[:-1] + [ops[1][-1]]

    if output_shape[-2] > tile_height and tile_height < TILE_DIM:
        output_shape[-2] = tile_height
    else:
        output_shape[-2] = align_up(output_shape[-2], tile_height)

    if is_sparse:
        assert len(attr) == 15, f"Unexpected number of attrs for sparse matmul: {len(attr)}"
        _, _, _, w, z, r, c, fracture_factor, _, _, _, _, t_factor_r, t_factor_c, _ = attr
        assert ops[1][-3] % z == 0

        # C dim should be multiplied by fracture_factor, but since we're doing a broadcast(C, fracture_factor), we also
        # need to divide C dim by fracture_factor, so they end up cancelling each other out
        return (w, t_factor_r * t_factor_c * z, align_up_tile(r) // t_factor_r // fracture_factor, align_up_tile(ops[1][3])), []

    broadcast = []
    #assert ops[0][0] == ops[1][0] # Relax this for now, although we really can't do W != 1 anyway.. so that needs fixing
    if ops[0][-3] != ops[1][-3]:
        if ops[0][-3] == 1:
            broadcast.append((0, 1, ops[1][-3]))
            output_shape[-3] = ops[1][-3]
        elif ops[1][-3] == 1:
            broadcast.append((1, 1, ops[0][-3]))
            output_shape[-3] = ops[0][-3]
        else:
            assert False, "If Z dimension is not the same for matmul, one of operands must have it be 1."

    if ops[0][-1] != ops[1][-2]:
        if ops[0][-1] == TILE_DIM:
            broadcast.append((0, 3, ops[1][-2]))
            output_shape[-1] = ops[1][-2]
        elif ops[1][-2] == TILE_DIM:
            broadcast.append((1, 2, ops[0][-1]))
            output_shape[-2] = ops[0][-1]
        else:
            print(ops)
            assert False, f"If inner dimension is not the same for matmul, one of operands must have it be {TILE_DIM}."

    if accumulate:
        output_shape[-3] = 1

    return tuple(output_shape), broadcast


def parallelization(type, attr, op_shape, fracture_factor):
    is_sparse = (len(attr) >= 2) and bool(attr[1])
    if is_sparse:
        assert op_shape.outputs[0].rt % fracture_factor == 0
        return (op_shape.outputs[0].rt // fracture_factor, op_shape.outputs[0].ct * fracture_factor)
    else:
        return (op_shape.outputs[0].rt, op_shape.outputs[0].ct)


def input_ublock_order(type, attr, num_operands):
    is_sparse = (len(attr) >= 2) and bool(attr[1])
    if is_sparse:
        return [UBlockOrder.C, UBlockOrder.R, UBlockOrder.R, UBlockOrder.R]
    return [UBlockOrder.C, UBlockOrder.R, UBlockOrder.R, UBlockOrder.R]


def execution_cycles(type, arch_name, op_model, theoretical) -> int:
    op_model_desc = op_model_to_desc(type, arch_name, op_model)

    compiler_cache_cycles = get_compiler_cached_cycles(op_model_desc)
    if compiler_cache_cycles is not None:
        return compiler_cache_cycles

    is_sparse = op_model.is_sparse_matmul
    if is_sparse:
        return get_op_model_execution_cycles(op_model_desc)

    # Math fidelity and data format are just estimated guesses for now
    math_fid = math_fidelity_to_multiplier(op_model.math_fidelity())
    u_kt = op_model.input_buffers[0].block_shape.ublock.ct
    m_k = op_model.op_shape.inputs[0].ct // u_kt
    t = op_model.output_buffers[0].block_shape.t
    mblock_m = op_model.output_buffers[0].block_shape.mblock_m
    mblock_n = op_model.output_buffers[0].block_shape.mblock_n
    ublock_rt = op_model.output_buffers[0].block_shape.ublock.rt
    ublock_ct = op_model.output_buffers[0].block_shape.ublock.ct

    mblock_executions = m_k * mblock_m * mblock_n
    ublock_executions = mblock_executions * u_kt * ublock_rt * ublock_ct

    is_cyclenet = "PYBUDA_CYCLENET" in os.environ
    if is_cyclenet:
        input0_df = data_format_to_int(op_model.input_buffers[0].data_format)
        output_df = data_format_to_int(op_model.output_buffers[0].data_format)
        x = [input0_df,output_df,math_fid,t,mblock_m,mblock_n,ublock_rt,ublock_ct,m_k,u_kt,mblock_executions,ublock_executions,0]
        cycle_count = cyclenet_execution_cycles(type, torch.tensor(x, dtype=torch.float32))
    elif theoretical:
        tile_weight = 32 if arch_name == 'grayskull' else 16
        cycle_count = t * ublock_executions * math_fid * tile_weight  # based on max throughput for the chip
    else:
        cycle_count = get_op_model_execution_cycles(op_model_desc)

        if op_model.input_buffers[0].data_format == DataFormat.Int8:
            if op_model.buda_op_attrs().get("bias") is True:
                op_model_desc.type = "nop"
                op_model_desc.mblock_k = 0
                op_model_desc.ublock_kt = 0
                cycle_count += get_op_model_execution_cycles(op_model_desc)

            if op_model.buda_op_attrs().get("requant") is True:
                op_model_desc.type = "requantization"
                op_model_desc.mblock_k = 0
                op_model_desc.ublock_kt = 0
                op_model_desc.math_fidelity = MathFidelity.HiFi4
                cycle_count += get_op_model_execution_cycles(op_model_desc)

            if op_model.buda_op_attrs().get("dequant") is True:
                op_model_desc.type = "dequantization"
                op_model_desc.mblock_k = 0
                op_model_desc.ublock_kt = 0
                op_model_desc.math_fidelity = MathFidelity.HiFi4
                cycle_count += get_op_model_execution_cycles(op_model_desc)

    return cycle_count

def cyclenet_execution_cycles(type, X) -> int:
    filepath = os.path.abspath(__file__)
    modelpath = os.path.join(os.path.dirname(filepath), f"{type}.pth")
    model = CycleNet(13, 64, 1)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device("cpu")))
    with torch.no_grad():
        model.eval()
        y_pred = model(X)
        total_cycles = y_pred.squeeze().cpu().numpy()
    return total_cycles
