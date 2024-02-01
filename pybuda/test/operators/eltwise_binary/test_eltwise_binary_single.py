# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import numpy as np

import pybuda
import pybuda.op
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import models

MODELS_PATH = "./pybuda/test/operators/eltwise_binary/models/"

def test_eltwise_binary_single(
    bin_model,
    bin_train,
    bin_recompute,
    bin_op,
    bin_shape
):

    print("\n")
    print(f"bin_train --> {bin_train}")
    print(f"bin_recompute --> {bin_recompute}")
    print(f"bin_op --> {bin_op}")
    print(f"bin_model --> {bin_model}")
    print(f"bin_shape --> {bin_shape}")
    print("\n")

    if not bin_train and bin_recompute:
        pytest.skip("Inference and recompute is the same as just inference.")
    
    assert type(bin_train) in [bool, str], "Type of training parameter must be boolean or string"
    if type(bin_train) == str:
        training = True if bin_train == 'True' else False
    else:
        training = bin_train
    assert type(bin_recompute) in [bool, str], "Type of recompute parameter must be boolean or string"
    if type(bin_recompute) == str:
        recompute = True if bin_recompute == 'True' else False
    else:
        recompute = bin_recompute
    operation = bin_op
    model = bin_model
    shape = eval(bin_shape) if type(bin_shape) == str else bin_shape

    print("\n")
    print(f"Training --> {training}")
    print(f"Recompute --> {recompute}")
    print(f"Operation --> {operation}")
    print(f"Model --> {model}")
    print(f"Shape --> {shape}")
    print("\n")

    architecture = f'models.{model}.BudaElementWiseBinaryTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape})'
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
