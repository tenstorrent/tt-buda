# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of pad operator
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig
from pybuda.config import CompileDepth, _get_global_compiler_config

from . import models

MODELS_PATH = "./pybuda/test/operators/tm/pad/models"

SHAPE_NO = 5
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 7
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 2
SHAPE_ZDIM_MAX = 2 ** 2

WDIM_FIXED = True
ZDIM_FIXED = False

PAD_NO = 5
PAD_MIN = 2 ** 3
PAD_MAX = 2 ** 6

PAD_SIZE_2 = 2
PAD_SIZE_4 = 4

np.random.seed(22)

pad = []
for _ in range(PAD_NO):
    if np.random.rand() > 0.5:
        padding = np.random.randint(PAD_MIN, PAD_MAX, (PAD_SIZE_2, )).tolist()
    else:
        padding = np.random.randint(PAD_MIN, PAD_MAX, (PAD_SIZE_4, )).tolist()
    pad.append(padding)


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

@pytest.mark.parametrize("pad", pad, ids=["pad" + "x".join([str(jtem) for jtem in item]) for item in pad])
@pytest.mark.parametrize("shape", shape, ids=["shape" + "x".join([str(jtem) for jtem in item]) for item in shape])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("mode", ["Inference"])
def test_hstack_hslice(
    test_device,
    mode,
    recompute,
    model, 
    shape,
    pad
):

    training = (mode == "Training")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaPadTest(shape={shape}, pad={pad})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute,
                        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER, # some reshapes decomposed into unsupported MMs
                     ), 
        verify_cfg=VerifyConfig()
    )
