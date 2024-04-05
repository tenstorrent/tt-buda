# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of reshape operator
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

MODELS_PATH = "./pybuda/test/operators/tm/reshape/models"

SHAPE_NO = 2
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4
SHAPE_DIM_MIN = 1
SHAPE_DIM_MAX = 2 ** 3
FACTOR = 2

TENSOR_FIXED = True
TENSOR_SIZE = 4

np.random.seed(1)

shape_factors =[np.random.randint(
    SHAPE_DIM_MIN, 
    SHAPE_DIM_MAX, 
    size=[
        (
            TENSOR_SIZE if TENSOR_FIXED else 
            np.random.randint(
                SHAPE_SIZE_MIN, 
                SHAPE_SIZE_MAX
            )
        ) * 2]) for _ in range(SHAPE_NO)]
old_shape = [np.prod(item.reshape(2, -1), 0).tolist() for item in  shape_factors]
new_shape = [np.prod(np.random.permutation(item).reshape(2, -1), 0).tolist() for item in  shape_factors]

@pytest.mark.parametrize("old_shape, new_shape", zip(old_shape, new_shape), ids=["old_shape=" + "x".join([str(item) for item in old]) + "-new_shape=" + "x".join([str(item) for item in new]) for old, new in zip(old_shape, new_shape)])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def test_reshape(
    op_test_kind,
    model, 
    old_shape, 
    new_shape
):

    test_kind = op_test_kind
    
    if model == "model_3":
        pytest.skip("These models return intermediate nodes. That's not supported today." 
                    "Autograd is trying to do backward pass twice for the same subpath in the graph and that's not correct. ")

    architecture = f'models.{model}.BudaReshapeTest(old_shape={old_shape}, new_shape={new_shape})'
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
