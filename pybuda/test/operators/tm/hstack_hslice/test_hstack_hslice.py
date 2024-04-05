# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of hstack, and hslice operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import models

TILE_DIM = 32

MODELS_PATH = "./pybuda/test/operators/tm/hstack_hslice/models"

SHAPE_NO = 1
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 3
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 2
SHAPE_ZDIM_MAX = 2 ** 2

SLICE_SIZE_MIN = 2 ** 3
SLICE_SIZE_MAX = 2 ** 7
SLICE_MIN = 2
SLICE_MAX = 2 ** 4

WDIM_FIXED = True

np.random.seed(1)

slices = [np.random.randint(SLICE_MIN, SLICE_MAX) for i in range(SHAPE_NO)]
shape = []
for i in range(SHAPE_NO):
    # ... create dimensions ...
    W = 1 if WDIM_FIXED else np.random.randint(SHAPE_WDIM_MIN, SHAPE_WDIM_MAX)
    Z = slices[i] * np.random.randint(SHAPE_ZDIM_MIN, SHAPE_ZDIM_MAX)
    R = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX)
    C = slices[i] * np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX) * TILE_DIM

    # ... create final test shape ...
    sh = [W, Z, R, C]
    shape.append(sh)

@pytest.mark.xfail
@pytest.mark.parametrize("shape, slice", zip(shape, slices), ids=["shape=" + "x".join([str(item) for item in sh])+ "-slice=" + str(sl) for sh, sl in zip(shape, slices)])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Inference"])
def test_hstack_hslice(
    mode,
    recompute,
    model, 
    shape, 
    slice
):

    training = (mode == "Training")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaHStackHSliceTest(shape={shape}, slice={slice})'
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
