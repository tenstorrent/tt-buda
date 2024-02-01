# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of layernorm layer
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import py
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import models

MODELS_PATH = "./pybuda/test/nn/layers/normalization/models/"

SHAPE_NO = 1
SHAPE_SIZE = 4

WDIM_FIXED = True
ZDIM_FIXED = True   # For now, layernorm can be applied only on trensor that hase z dimension equals to 1

# Dimension sizes controlled by above defiend flags
SHAPE_WDIM_MIN = 2
SHAPE_WDIM_MAX = 2 ** 6
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 8

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 6

EPSILON_NO = 3
EPSILON_MIN = 1e-6
EPSILON_MAX = 1e-4

np.random.seed(2)

shape_data = []
for _ in range(SHAPE_NO):

    # ... w dimension, batch ...
    wdim = 1 if WDIM_FIXED else np.random.randint(SHAPE_WDIM_MIN, SHAPE_WDIM_MAX)
    # ... z dimension, channels ...
    zdim = 1 if ZDIM_FIXED else np.random.randint(SHAPE_ZDIM_MIN, SHAPE_ZDIM_MAX)
    # ... r, c dimensions ...
    rdim = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX)
    cdim = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX)

    # ... add dimensions ...
    input_shape = [wdim, zdim, rdim, cdim]
    gamma_shape = [1, 1, 1, cdim]
    beta_shape = [1, 1, 1, cdim]
    shape_data.append([input_shape, gamma_shape, beta_shape])

shape_ids = [f"shape-input{'x'.join([str(jtem) for jtem in item[0]])}-"
             f"gamma{'x'.join([str(jtem) for jtem in item[1]])}-"
             f"beta{'x'.join([str(jtem) for jtem in item[2]])}"  for item in shape_data]

print(f"shape data length: {len(list(shape_data))}")
print(f"shape ids length: {len(shape_ids)}")

epsilon = np.random.random_sample([EPSILON_NO, ]) * (EPSILON_MAX - EPSILON_MIN) + EPSILON_MIN


@pytest.mark.parametrize("epsilon", epsilon)
@pytest.mark.parametrize("shape", shape_data, ids=shape_ids)
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("mode", ["Training", "Inference"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
def test_layernorm(
    mode,
    recompute,
    model,
    shape,
    dim,
    epsilon
):

    training = (mode == "Training")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    input_shape, gamma_shape, beta_shape = shape

    architecture = f'models.{model}.LayernormTest(input_shape={input_shape}, gamma_shape={gamma_shape}, beta_shape={beta_shape}, dim={dim}, epsilon={epsilon})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)

    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute,
                     ), 
        verify_cfg=VerifyConfig()
    )