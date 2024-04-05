# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of leaky relu operator
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

MODELS_PATH = "./pybuda/test/operators/eltwise_unary_attr/leaky_relu/models"

SHAPE_NO = 3
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 6
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 6
SHAPE_ZDIM_MIN = 2
SHAPE_ZDIM_MAX = 2 ** 6

WDIM_FIXED = True
ZDIM_FIXED = True

ALPHA_NO = 3
ALPHA_MIN = 1.0
ALPHA_MAX = 5.0

np.random.seed(3)

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

alpha = np.random.rand(ALPHA_NO) * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN


@pytest.mark.parametrize("alpha", alpha, ids=[f"alpha={item}" for item in alpha])
@pytest.mark.parametrize("shape", shape, ids=["shape=" + "x".join([str(item) for item in sh]) for sh in shape])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Inference"])
def test_leaky_relu(
    mode,
    recompute,
    model, 
    shape,
    alpha
):

    training = (mode == "Training")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaLeakyReluTest(shape={shape}, alpha={alpha})'
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
