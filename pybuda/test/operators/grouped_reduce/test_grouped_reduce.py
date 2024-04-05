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
import torch

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from pybuda.verify.config import TestKind

from . import models

MODELS_PATH = "./pybuda/test/operators/grouped_reduce/models/"

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

from functools import reduce
def factors(n):    
    facs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            facs.add(i)
            facs.add(n//i)
    return sorted(list(facs))

torch.manual_seed(0)
shapes = [[1, int(torch.randint(1, 8, (1,))), int(torch.randint(1, 120, (1,))), int(torch.randint(1, 120, (1,)))] for _ in range(20)] + [[1, 1, 1, 9]]
@pytest.mark.parametrize("shape", shapes, ids=[f"shape={'x'.join([str(jtem) for jtem in item])}" for item in shapes])
@pytest.mark.parametrize("operation", ["GroupedReduceAvg"])
@pytest.mark.parametrize("model", [item.split(".")[0] for item in os.listdir(MODELS_PATH) if "model" in item])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("keep_dims", [True, False])
@pytest.mark.parametrize("op_test_kind", [TestKind.INFERENCE])
def test_grouped_reduce(
    op_test_kind,
    operation,
    model,
    shape,
    dim,
    keep_dims,
):
    test_kind = op_test_kind
    facs = factors(shape[dim])
    if len(facs) < 3:
        pytest.skip("Not enough factors")
    facs = facs[1:-1]
    
    groups_to_try = [np.random.choice(facs) for _ in range(min(len(facs), 3))]
    # groups_to_try = [8]
    for groups in groups_to_try:
        architecture = f'models.{model}.BudaReduceTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape}, dim={dim}, groups={groups}, keep_dims={keep_dims})'
        tt_model = eval(architecture)
        tt0 = TTDevice("tt0", devtype=BackendType.Golden)
        tt0.place_module(tt_model)
        pybuda_compile(
            tt0, 
            tt_model.testname, 
            *tt_model.inputs, 
            compiler_cfg=CompilerConfig(
                            enable_training=test_kind.is_training(),
                            enable_recompute=test_kind.is_recompute()
                        ), 
            verify_cfg=VerifyConfig()
        )
