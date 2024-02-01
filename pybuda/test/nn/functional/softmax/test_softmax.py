# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of softmax operator
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

MODELS_PATH = "./pybuda/test/nn/functional/softmax/models"

SHAPE_NO = 5
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 7
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True
SIZE_FIXED = False

np.random.seed(1)

# tensors of arbitrarily shape
# shape = [np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, np.random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)).tolist() for _ in range(SHAPE_NO)]
shape = []
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
dim = [np.random.randint(len(sh) - 2, len(sh)) for sh in shape]


@pytest.mark.parametrize("stable", (True, False), ids=["StableSoftmax", "OriginalSoftmax"])
@pytest.mark.parametrize("shape, dim", zip(shape, dim), ids=["shape=" + "x".join([str(item) for item in sh]) + "-dim=" + str(dm) for sh, dm in zip(shape, dim)])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Training", "Inference"])
def test_softmax(
    mode,
    recompute,
    model, 
    shape, 
    dim,
    stable
):

    training = (mode == "Training")

    # if model in ["model_4", "model_5"]:
    #     pytest.skip("These models return intermediate nodes. That's not supported today." 
    #                 "Autograd is trying to do backward pass twice for the same subpath in the graph and that's not correct. ")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaSoftmaxTest(shape={shape}, dim={dim}, stable={stable})'
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
