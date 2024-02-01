# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import pybuda
import pybuda.op
from pybuda import (
    PyBudaModule,
    TTDevice,
    BackendType,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
)

SIZE_1K = 1024


shape = [

    # X = 1GB
    [1, 1, 512, SIZE_1K * SIZE_1K],
    [1, 1, 256, 2 * SIZE_1K * SIZE_1K],
    [1, 1, 128, 4 * SIZE_1K * SIZE_1K],

    [1, 1, SIZE_1K * SIZE_1K, 512],
    [1, 1, 2 * SIZE_1K * SIZE_1K, 256],
    [1, 1, 4 * SIZE_1K * SIZE_1K, 128],


    # 1GB < X <= 2GB
    [1, 1, SIZE_1K, SIZE_1K * SIZE_1K],
    [1, 1, 512, 2 * SIZE_1K * SIZE_1K],
    [1, 1, 768, 2 * SIZE_1K * SIZE_1K],
    [1, 1, 256, 4 * SIZE_1K * SIZE_1K],

    [1, 1, SIZE_1K * SIZE_1K, SIZE_1K],
    [1, 1, 2 * SIZE_1K * SIZE_1K, 512],
    [1, 1, 4 * SIZE_1K * SIZE_1K, 256],
    [1, 1, 2 * SIZE_1K * SIZE_1K, 768],


    # 2GB < X <= 4GB
    [1, 1, 2 * SIZE_1K, SIZE_1K * SIZE_1K],
    [1, 1, SIZE_1K, 2 * SIZE_1K * SIZE_1K],
    [1, 1, 512, 4 * SIZE_1K * SIZE_1K],
    [1, 1, 256, 8 * SIZE_1K * SIZE_1K],
    [1, 1, 1536, SIZE_1K * SIZE_1K],
    [1, 1, 768, 2 * SIZE_1K * SIZE_1K],

    [1, 1, SIZE_1K * SIZE_1K, 2 * SIZE_1K],
    [1, 1, SIZE_1K * SIZE_1K, 2 * SIZE_1K],
    [1, 1, 512 * SIZE_1K, 4 * SIZE_1K],
    [1, 1, 256 * SIZE_1K, 8 * SIZE_1K],
    [1, 1, 1536 * SIZE_1K, SIZE_1K],
    [1, 1, 768 * SIZE_1K, 2 * SIZE_1K],

    # 4GB < X <= 8GB
    [1, 1, 4 * SIZE_1K, SIZE_1K * SIZE_1K],
    [1, 1, SIZE_1K * SIZE_1K, 4 * SIZE_1K]
]



class ElementWiseBinary(PyBudaModule):
    
    def __init__(self, name, weight_shape):
        super().__init__(name)
        self.shape = weight_shape
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act):
        return pybuda.op.Multiply("mul", act, self.weights1)

@pytest.mark.xfail(
    reason="tenstorrent/pybuda#25"
)
@pytest.mark.parametrize("shape", shape, ids=[f"shape={'x'.join([str(item) for item in sh])}" for sh in shape])
def test_eltwise_binary(shape):

    model = ElementWiseBinary("Large Parameter Binary", shape)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    activations = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    pybuda_compile(tt0, model.name, activations)