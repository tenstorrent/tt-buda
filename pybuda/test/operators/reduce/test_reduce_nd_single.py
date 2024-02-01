# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of reduce operators
#
# In this test we use pytorch tensors and operators to verify buda operators
#

import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import models_nd

MODELS_PATH = "./pybuda/test/operators/reduce/models_nd/"

@pytest.mark.xfail
def test_reduce(
    red_train,
    red_recompute,
    red_op,
    red_model,
    red_shape,
    red_dim,
    red_keepdim
):

    print("\n")
    print(f"red_train --> {red_train}")
    print(f"red_recompute --> {red_recompute}")
    print(f"red_op --> {red_op}")
    print(f"red_model --> {red_model}")
    print(f"red_shape --> {red_shape}")
    print(f"red_dim --> {red_dim}")
    print(f"red_keepdim --> {red_keepdim}")
    print("\n")

    if not red_train and red_recompute:
        pytest.skip("Inference and recompute is the same as just inference.")
    
    assert type(red_train) in [bool, str], "Type of training parameter must be boolean or string"
    if type(red_train) == str:
        training = True if red_train == 'True' else False
    else:
        training = red_train
    assert type(red_recompute) in [bool, str], "Type of recompute parameter must be boolean or string"
    if type(red_recompute) == str:
        recompute = True if red_recompute == 'True' else False
    else:
        recompute = red_recompute
    model = red_model
    shape = eval(red_shape) if type(red_shape) == str else red_shape
    operation = red_op
    dim = int(red_dim)
    keepdim = True if red_keepdim == 'True' else False

    print("\n")
    print(f"Training --> {training}")
    print(f"Recompute --> {recompute}")
    print(f"Operation --> {operation}")
    print(f"Model --> {model}")
    print(f"Shape --> {shape}")
    print(f"Dim --> {dim}")
    print(f"Keepdim --> {keepdim}")
    print("\n")

    architecture = f'models_nd.{model}.BudaReduceTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape}, dim={dim}, keepdim={keepdim})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute
                     ), 
        verify_cfg=VerifyConfig()
    )