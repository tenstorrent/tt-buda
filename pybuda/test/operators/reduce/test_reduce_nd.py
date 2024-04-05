# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of reduce operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import models_nd

MODELS_PATH = "./pybuda/test/operators/reduce/models_nd/"

SHAPE_NO = 1
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 2
SHAPE_DIM_MAX = 2 ** 5
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True

np.random.seed(20)

# tensors of arbitrarily shape
shape = [np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, np.random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)).tolist() for _ in range(SHAPE_NO)]
# 4-dimensional tensors
if SHAPE_FIXED:
    size = 4
    shape_fixed = []
    for _ in range(SHAPE_NO):
        wdim = 1 if WDIM_FIXED else np.random.randint(SHAPE_WDIM_MIN, SHAPE_WDIM_MAX)
        zdim = np.random.randint(SHAPE_ZDIM_MIN, SHAPE_ZDIM_MAX)
        sh = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, 2).tolist()
        shape_fixed.append([wdim, zdim] + sh)
    shape += shape_fixed

dim = [np.random.randint(0, len(sh) - 1) for sh in shape]

@pytest.mark.xfail(
    reason="Not Implemented"
)
@pytest.mark.parametrize("keepdim", [True, False], ids=["KeepDim", "NoKeepDim"])
@pytest.mark.parametrize("shape, dim", zip(shape, dim), ids=[f"shape={'x'.join([str(item) for item in sh])}-dim={dm}" for sh, dm in zip(shape, dim)])
@pytest.mark.parametrize("operation", ["ReduceSum", "ReduceAvg"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Inference"])
def test_reduce(
    mode,
    recompute,
    operation,
    model,
    shape,
    dim,
    keepdim
):

    training = (mode == "Training")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models_nd.{model}.BudaReduceTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape}, dim={dim}, keepdim={keepdim})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute
                     ), 
        verify_cfg=VerifyConfig()
    )
