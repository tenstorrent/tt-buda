# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import random
import pybuda
from pybuda import (
    Tensor,
    CompilerConfig,
    VerifyConfig,
)
from .common import compile, run
from pybuda.pybudaglobal import TILE_DIM


@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize(
    "test",
    [
        ((1, 128, 16, 256), 3, 0, 64, 1),
        ((1,   2, 16, 256), 3, 0, 64, 2),
        ((1, 128, 16, 256), 3, 1, 64, 2),
        ((1, 128, 16, 256), 3, 64, 256, 1),
        ((1,  16,  5,   5), 1, 0, 1, 16),
        ((1,  16,  5,   5), 1, 2, 9, 2),
        ((3, 265, 124), 0, 1, 3, 2),
    ],
)
def test_index(
    mode,
    test,
):
    training = mode == "training"
    shape, dim, start, stop, stride = test
    if start == 2 and stop == 9 and training: # test5-training
        pytest.skip("temporarily skip it, fails on pipeline but cannot reproduce it locally")

    @compile(
        compiler_cfg=CompilerConfig(enable_training=training, enable_t_streaming=True),
        verify_cfg=VerifyConfig(),
    )
    def index(x):
        return pybuda.op.Index("index", x, dim, start, stop, stride)

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=training))
    index(x)


@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize(
    "test",
    [
        ((64,), (0, 32, 1)),
        ((64,), (0, 64, 2)),
        ((64,), (3, 32, 2)),
        ((59,), (17, 37, 2)),
        ((2049,), (1, 2000, 9)),
    ],
)
def test_index1d(
    mode,
    test,
):
    training = mode == "training"

    shape, index = test
    start, stop, stride = index
    dim = 0

    @compile(
        compiler_cfg=CompilerConfig(enable_training=training),
        verify_cfg=VerifyConfig(),
    )
    def index1d(x):
        return pybuda.op.Index("index", x, dim, start, stop, stride)

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=training))
    index1d(x)

@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize(
    "test",
    [
        ((43, 17), (1, 40, 2), (2, 10, 3)),
        1,
        2,
        3,
        4,
        5,
    ],
)
def test_index2d(
    mode,
    test,
):
    training = mode == "training"

    if type(test) is int:
        random.seed(test)
        r = random.randint(4, 512)
        c = random.randint(4, 512)
        shape = (r, c)
        start0 = random.randint(0, r - 1)
        stop0 = random.randint(start0 + 1, r)
        stride0 = random.randint(1, stop0 - start0)
        start1 = random.randint(0, c - 1)
        stop1 = random.randint(start1 + 1, c)
        stride1 = random.randint(1, stop1 - start1)
    else:
        shape, index0, index1 = test
        start0, stop0, stride0 = index0
        start1, stop1, stride1 = index1

    @compile(
        compiler_cfg=CompilerConfig(enable_training=training),
        verify_cfg=VerifyConfig(),
    )
    def index2d(x):
        x = pybuda.op.Index("index_r", x, 0, start0, stop0, stride0)
        x = pybuda.op.Index("index_c", x, 1, start1, stop1, stride1)
        return x

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=training))
    index2d(x)

@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize(
    "test",
    [
        1,
        2,
        3,
        4,
        5,
    ],
)
def test_index3d(
    mode,
    test,
):
    training = mode == "training"

    if training:
        pytest.skip("tenstorrent/pybuda#184")

    if type(test) is int:
        random.seed(test)
        has_w = random.randint(0, 1)
        z = random.randint(1, 16)
        r = random.randint(4, 512)
        c = random.randint(4, 512)
        if has_w:
            shape = (1, z, r, c)
        else:
            shape = (z, r, c)
        start0 = random.randint(0, z - 1)
        stop0 = random.randint(start0 + 1, z)
        stride0 = random.randint(1, stop0 - start0)
        start1 = random.randint(0, r - 1)
        stop1 = random.randint(start1 + 1, r)
        stride1 = random.randint(1, stop1 - start1)
        start2 = random.randint(0, c - 1)
        stop2 = random.randint(start2 + 1, c)
        stride2 = random.randint(1, stop2 - start2)
    else:
        shape, index0, index1, index2 = test
        has_w = len(shape) == 4
        start0, stop0, stride0 = index0
        start1, stop1, stride1 = index1
        start2, stop2, stride2 = index2

    @compile(
        compiler_cfg=CompilerConfig(enable_training=training, enable_t_streaming=True),
        verify_cfg=VerifyConfig(),
    )
    def index3d(x):
        x = pybuda.op.Index("index_z", x, has_w + 0, start0, stop0, stride0)
        x = pybuda.op.Index("index_r", x, has_w + 1, start1, stop1, stride1)
        x = pybuda.op.Index("index_c", x, has_w + 2, start2, stop2, stride2)
        return x

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=training))
    index3d(x)