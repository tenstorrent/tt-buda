# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of hstack, and hslice operators
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

MODELS_PATH = "./pybuda/test/operators/tm/hstack_hslice/models"

def test_hstack_hslice(
    tmh_train,
    tmh_recompute,
    tmh_model, 
    tmh_shape, 
    tmh_slice
):

    print("\n")
    print(f"tmh_train --> {tmh_train}")
    print(f"tmh_recompute --> {tmh_recompute}")
    print(f"tmh_model --> {tmh_model}")
    print(f"tmh_shape --> {tmh_shape}")
    print(f"tmh_slice --> {tmh_slice}")
    print("\n")

    if not tmh_train and tmh_recompute:
        pytest.skip("Inference and recompute is the same as just inference.")
    
    assert type(tmh_train) in [bool, str], "Type of training parameter must be boolean or string"
    if type(tmh_train) == str:
        training = True if tmh_train == 'True' else False
    else:
        training = tmh_train
    assert type(tmh_recompute) in [bool, str], "Type of recompute parameter must be boolean or string"
    if type(tmh_recompute) == str:
        recompute = True if tmh_recompute == 'True' else False
    else:
        recompute = tmh_recompute
    model = tmh_model
    shape = eval(tmh_shape) if type(tmh_shape) == str else tmh_shape
    slice = int(tmh_slice)

    print("\n")
    print(f"Training --> {training}")
    print(f"Recompute --> {recompute}")
    print(f"Model --> {model}")
    print(f"Shape --> {shape}")
    print(f"Slice --> {slice}")
    print("\n")


    architecture = f'models.{model}.BudaHStackHSliceTest(shape={shape}, slice={slice})'
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
