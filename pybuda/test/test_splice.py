# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import yaml
import random
import time
import os

import pybuda
from pybuda.op.eval.buda.splice import Splice
from pybuda.op.eval.buda.tm import eval as tm_eval
from pybuda._C.balancer import OpShape
from pybuda.pybudaglobal import TILE_DIM


def factorize(n):
    import numpy as np

    factors = np.arange(1, n + 1)
    factors = factors[n % factors == 0]
    return list(factors)


def _gen_splice_pars(par):
    r_factors = factorize(par[0])
    c_factors = factorize(par[1])
    for ublock_order_r in [True, False]:
        for ublock_r in r_factors:
            for ublock_c in c_factors:
                for grid_r in r_factors:
                    for grid_c in c_factors:
                        for t_r in r_factors:
                            for t_c in c_factors:
                                if par[0] % (t_r * grid_r * ublock_r) != 0:
                                    continue
                                if par[1] % (t_c * grid_c * ublock_c) != 0:
                                    continue
                                yield ublock_order_r, ublock_r, ublock_c, grid_r, grid_c, t_r, t_c


def stream(tensor, t_stream_factor_r, t_stream_factor_c):
    if type(tensor) is list:
        return [stream(t, t_stream_factor_r, t_stream_factor_c) for t in tensor]
    result = tensor
    if t_stream_factor_r > 1:
        result = tm_eval("vslice", [t_stream_factor_r], [result])
    if t_stream_factor_c > 1:
        result = tm_eval("hslice", [t_stream_factor_c], [result])
    return result


def unstream(tensor, t_stream_factor_r, t_stream_factor_c):
    if type(tensor) is list:
        return [unstream(t, t_stream_factor_r, t_stream_factor_c) for t in tensor]
    result = tensor
    if t_stream_factor_c > 1:
        result = tm_eval("hstack", [t_stream_factor_c], [result])
    if t_stream_factor_r > 1:
        result = tm_eval("vstack", [t_stream_factor_r], [result])
    return result


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("convert_mode_t", [False, True])
def test_concat(dim, convert_mode_t):
    if dim == 1 and convert_mode_t:
        # splice on z dim cannot convert to mode t
        pytest.skip()

    num_tiles = 12 if dim > 1 else 4
    tensors = [torch.randn(1, 1, 32 * num_tiles, 32 * num_tiles)] * 3
    tensor_shapes = [tuple(t.shape) for t in tensors]
    golden = torch.cat(tensors, dim=dim)

    splice = Splice.create_concatenate(dim, tensor_shapes)
    result = splice.eval(tensors)
    assert torch.allclose(result, golden)

    output_shape = splice.shape(tensor_shapes, TILE_DIM, TILE_DIM)[0]
    if convert_mode_t:
        orig_dim, input_slices, output_stack = splice.convert_mode_t()
        output_shape = splice.shape(tensor_shapes, TILE_DIM, TILE_DIM)[0]

    par = splice.parallelization(OpShape(tensor_shapes, tuple(output_shape)))
    for (
        ublock_order_r,
        ublock_r,
        ublock_c,
        grid_r,
        grid_c,
        t_stream_factor_r,
        t_stream_factor_c,
    ) in _gen_splice_pars(par):
        splice.update_ranges(
            ublock_order_r,
            ublock_r,
            ublock_c,
            grid_r,
            grid_c,
            t_stream_factor_r,
            t_stream_factor_c,
        )
        result = tensors
        if convert_mode_t:
            dir_r = orig_dim == 2
            result = [
                stream(
                    r, input_slices[i] if dir_r else 1, 1 if dir_r else input_slices[i]
                )
                for i, r in enumerate(result)
            ]
        result = stream(result, t_stream_factor_r, t_stream_factor_c)
        result = splice.eval(result)
        result = unstream(result, t_stream_factor_r, t_stream_factor_c)
        if convert_mode_t:
            dir_r = orig_dim == 2
            result = unstream(
                result, output_stack if dir_r else 1, 1 if dir_r else output_stack
            )
        assert torch.allclose(
            result, golden
        ), f"ublock_order_r={ublock_order_r}, ublock_r={ublock_r}, ublock_c={ublock_c}, grid_r={grid_r}, grid_c={grid_c}, t_stream_factor_r={t_stream_factor_r}, t_stream_factor_c={t_stream_factor_c}"
