# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of clip operator
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

MODELS_PATH = "./pybuda/test/operators/eltwise_unary_attr/clip/models"

SHAPE_NO = 2
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 10
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 6
SHAPE_ZDIM_MIN = 2
SHAPE_ZDIM_MAX = 2 ** 6

WDIM_FIXED = True
ZDIM_FIXED = True

MIN_VALUE_NO = 5
MAX_VALUE_NO = 5
MIN_VALUE_MIN = 0.0
MIN_VALUE_MAX = 0.5
MAX_VALUE_MIN = 0.5
MAX_VALUE_MAX = 1.0

np.random.seed(1)

shape = []
for i in range(SHAPE_NO):
    # ... create dimensions ...
    W = 1 if WDIM_FIXED else np.random.randint(SHAPE_WDIM_MIN, SHAPE_WDIM_MAX)
    Z = 1 if ZDIM_FIXED else np.random.randint(SHAPE_ZDIM_MIN, SHAPE_ZDIM_MAX)
    R = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX)
    C = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX)

    # ... create final test shape ...
    sh = [W, Z, R, C]
    shape.append(sh)

min_value = np.random.rand(MIN_VALUE_NO) * (MIN_VALUE_MAX - MIN_VALUE_MIN) + MIN_VALUE_MIN 
max_value = np.random.rand(MAX_VALUE_NO) * (MAX_VALUE_MIN - MAX_VALUE_MIN) + MAX_VALUE_MIN


@pytest.mark.parametrize("max_value", max_value, ids=[f"max={item}" for item in max_value])
@pytest.mark.parametrize("min_value", min_value, ids=[f"min={item}" for item in min_value])
@pytest.mark.parametrize("shape", shape, ids=["shape=" + "x".join([str(item) for item in sh]) for sh in shape])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Inference"])
def test_clip(
    mode,
    recompute,
    model, 
    shape,
    min_value,
    max_value
):

    training = (mode == "Training")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaClipTest(shape={shape}, min_value={min_value}, max_value={max_value})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)

    #Fusing disabled due to tenstorrent/pybuda#784
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute,
                        enable_auto_fusing=False
                     ), 
        verify_cfg=VerifyConfig()
    )
