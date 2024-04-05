# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of reduce operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#
# In these tests we suppose that all shapes are 4-dimensional and keepdim is always True
# (e.g, before reduce shape=(64, 32, 256, 128), after reduce over dimension 2 shape=(64, 32, 1, 128))
#

import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from pybuda.verify.config import TestKind

from . import models_4d

MODELS_PATH = "./pybuda/test/operators/reduce/models_4d/"

SHAPE_NO = 1
SHAPE_SIZE = 4
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 5

SHAPE_WDIM_MIN = 2
SHAPE_WDIM_MAX = 2 ** 3
SHAPE_ZDIM_MIN = 2
SHAPE_ZDIM_MAX = 2 ** 3

WDIM_FIXED = True

np.random.seed(100)

# 4-dimensional tensors
shape = []
for _ in range(SHAPE_NO):
    wdim = 1 if WDIM_FIXED else np.random.randint(SHAPE_WDIM_MIN, SHAPE_WDIM_MAX)
    zdim = np.random.randint(SHAPE_ZDIM_MIN, SHAPE_ZDIM_MAX)
    sh = np.random.randint(SHAPE_DIM_MIN, SHAPE_DIM_MAX, 2).tolist()
    shape.append([wdim, zdim] + sh)


@pytest.mark.parametrize("shape", shape, ids=[f"shape={'x'.join([str(jtem) for jtem in item])}" for item in shape])
@pytest.mark.parametrize("operation", ["ReduceSum", "ReduceAvg", "ReduceMax"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def test_reduce(
    op_test_kind,
    operation,
    model,
    shape
):

    test_kind = op_test_kind

    if operation == "ReduceMax" and test_kind.is_training():
        pytest.xfail()

    if test_kind.is_training() and len(shape) >= 3 and shape[-3] > 1:
        pytest.skip("Matmul with gradient accumulate must have t=1")

    if model in ["model_1", "model_5"]:
        pytest.skip("These models return intermediate nodes. That's not supported today." 
                    "Autograd is trying to do backward pass twice for the same subpath in the graph and that's not correct. ")

    architecture = f'models_4d.{model}.BudaReduceTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute()
                     ), 
        verify_cfg=VerifyConfig()
    )
