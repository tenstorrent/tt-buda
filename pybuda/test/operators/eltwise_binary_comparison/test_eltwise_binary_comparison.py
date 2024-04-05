# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise binary comparison operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

from ast import operator
import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import models

MODELS_PATH = "./pybuda/test/operators/eltwise_binary_comparison/models/"

SHAPE_NO = 1
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 10
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 2
SHAPE_ZDIM_MAX = 2 ** 2

SLICE_SIZE_MIN = 2 ** 3
SLICE_SIZE_MAX = 2 ** 7
SLICE_MIN = 2
SLICE_MAX = 2 ** 4

WDIM_FIXED = True
ZDIM_FIXED = True

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

rng_min = [-10.0, 0.0]
rng_max = [ 10.0, 1.0]
rng = list(zip(rng_min, rng_max))

# print(rng)

@pytest.mark.parametrize("rng_min, rng_max", rng, ids=[(f"min={item[0]}-max={item[1]}") for item in rng])
@pytest.mark.parametrize("mask", (True, False), ids=["Masked", "NotMasked"])
@pytest.mark.parametrize("op", ["Greater", "Less", "GreaterEqual", "LessEqual", "Equal", "NotEqual"])
@pytest.mark.parametrize("shape", shape, ids=[f"shape{'x'.join([str(jtem) for jtem in item])}" for item in shape])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Inference"])
def test_comparison(
    mode,
    recompute,
    model, 
    shape,
    op,
    mask,
    rng_min,
    rng_max
):

    training = (mode == "Training")

    if training:
        pytest.skip()

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaComparisonTest(' +\
                                        f'shape={shape} ,' +\
                                        f'opname="{op}" ,' +\
                                        f'operator=pybuda.op.{op} ,' +\
                                        f'mask={mask} ,' +\
                                        f'rng_min={rng_min} ,' +\
                                        f'rng_max={rng_max})'
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
