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

from . import models

MODELS_PATH = "./pybuda/test/operators/tm/reshape/models"

def test_reshape(
    resh_train,
    resh_recompute,
    resh_model, 
    resh_oshape, 
    resh_nshape
):

    print("\n")
    print(f"resh_train --> {resh_train}")
    print(f"resh_recompute --> {resh_recompute}")
    print(f"resh_model --> {resh_model}")
    print(f"resh_oshape --> {resh_oshape}")
    print(f"resh_nshape --> {resh_nshape}")
    print("\n")

    if not resh_train and resh_recompute:
        pytest.skip("Inference and recompute is the same as just inference.")

    assert type(resh_train) in [bool, str], "Type of training parameter must be boolean or string"
    if type(resh_train) == str:
        training = True if resh_train == 'True' else False
    else:
        training = resh_train
    assert type(resh_recompute) in [bool, str], "Type of recompute parameter must be boolean or string"
    if type(resh_recompute) == str:
        recompute = True if resh_recompute == 'True' else False
    else:
        recompute = resh_recompute
    model = resh_model
    old_shape = eval(resh_oshape) if type(resh_oshape) == str else resh_oshape
    new_shape = eval(resh_nshape) if type(resh_nshape) == str else resh_nshape

    print("\n")
    print(f"Training --> {training}")
    print(f"Recompute --> {recompute}")
    print(f"Model --> {model}")
    print(f"Old Shape --> {old_shape}")
    print(f"New Operation --> {new_shape}")
    print("\n")

    architecture = f'models.{model}.BudaReshapeTest(old_shape={old_shape}, new_shape={new_shape})'
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
