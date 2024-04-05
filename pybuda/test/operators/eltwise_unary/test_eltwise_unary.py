# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise unary operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import pytest
import numpy as np

import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig
from pybuda.verify.config import TestKind

from . import models

MODELS_PATH = "./pybuda/test/operators/eltwise_unary/models/"

SHAPE_NO = 2
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4

SHAPE_DIM_MIN = 2 ** 3
SHAPE_DIM_MAX = 2 ** 5
SHAPE_WDIM_MIN = 1
SHAPE_WDIM_MAX = 2 ** 2
SHAPE_ZDIM_MIN = 1
SHAPE_ZDIM_MAX = 2 ** 2

SHAPE_FIXED = True
WDIM_FIXED = True

np.random.seed(1)

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
@pytest.mark.parametrize("operation", ["Abs", "LeakyRelu", "Exp", "Identity", "Reciprocal", "Sigmoid", "Sqrt", "Gelu", "Log", "Relu", "Buffer", "Tanh", "Dropout", "Sine", "Cosine", "Argmax", "Clip"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def test_eltwise_unary(
    op_test_kind,
    operation,
    model,
    shape
):
    test_kind = op_test_kind

    if model == "model_9" and operation == "Reciprocal":
        pytest.xfail("tenstorrent/pybuda#18")

    kwargs = {}
    pcc = 0.99
    if operation == "LeakyRelu":
        kwargs['alpha'] = np.random.rand()
        if test_kind.is_training():
            pcc = 0.95
    if operation == "Clip":
        kwargs['min'] = np.random.rand()
        kwargs['max'] = np.random.rand()
        
    architecture = f'models.{model}.BudaElementWiseUnaryTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape}'
    for k, v in kwargs.items():
        architecture = f'{architecture}, {k}={v}'
    architecture = f'{architecture})'
    
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)

    #Fusing disabled due to tenstorrent/pybuda#784
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=test_kind.is_training(),
                        enable_recompute=test_kind.is_recompute(),
                        enable_auto_fusing=False
                     ), 
        verify_cfg=VerifyConfig(pcc=pcc)
    )
