# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of matmul operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from pybuda.verify.config import TestKind

from .models import generic
from .models import custom

MODELS_PATH = "./pybuda/test/operators/matmul/models/"
MODELS_GENERIC_PATH = MODELS_PATH + "generic/" 
MODELS_CUSTOM_PATH = MODELS_PATH + "custom/"

SHAPE_NO = 1
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

# Generic Shape

#@pytest.mark.xfail(
#    reason="tenstorrent/pybuda#22"
#)
@pytest.mark.parametrize("shape", shape, ids=[f"shape{'x'.join([str(jtem) for jtem in item])}" for item in shape])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_GENERIC_PATH) if "model" in item])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def test_matmul_generic(
    op_test_kind,
    model,
    shape
):
    test_kind = op_test_kind
    if test_kind.is_training() and len(shape) >= 3 and shape[-3] > 1:
        pytest.skip("Matmul with gradient accumulate must have t=1")

    architecture = f'generic.{model}.BudaMatmulTest(shape={shape})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    print(model.get_parameters())
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute(),
                     ), 
        verify_cfg=VerifyConfig()
    )


# Custom Shape

@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_CUSTOM_PATH) if "model" in item])
def test_matmul_custom(
    test_kind,
    model
):

    if test_kind.is_training():
        pytest.xfail() # numbers gets too big

    if model == "model_4":
        pytest.xfail() # balancer failure

    architecture = f'custom.{model}.BudaMatmulTest()'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute(),
                     ), 
        verify_cfg=VerifyConfig()
    )
