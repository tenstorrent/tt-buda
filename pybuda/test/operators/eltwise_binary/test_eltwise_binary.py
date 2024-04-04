# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise binary operators
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

MODELS_PATH = "./pybuda/test/operators/eltwise_binary/models/"

SHAPE_NO = 2
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 6
SHAPE_WDIM_MIN = 2
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True

np.random.seed(2)

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


@pytest.mark.parametrize("shape", shape, ids=[f"shape{'x'.join([str(jtem) for jtem in item])}" for item in shape])
@pytest.mark.parametrize("operation", ["Add", "Max", "Min", "Power", "Subtract", "Multiply", "Heaviside", "Greater", "GreaterEqual", "Less", "LessEqual", "Equal", "NotEqual"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("mode", ["Inference"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
def test_eltwise_binary(
    mode,
    recompute,
    operation,
    model,
    shape,
    test_device
):

    training = (mode == "Training")

    if training and (operation in ["Greater", "GreaterEqual", "Less", "LessEqual", "Equal", "NotEqual"]):
        pytest.skip("Comparison operators shouldn't have derivative, and backward.")

    if training and operation == "Heaviside":
        pytest.skip("Heaviside function shouldn't have derivative, and backward.")

    if not training and recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    architecture = f'models.{model}.BudaElementWiseBinaryTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)

    #Fusing disabled due to tenstorrent/pybuda#548
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
