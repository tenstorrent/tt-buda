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

from . import models

MODELS_PATH = "./pybuda/test/operators/eltwise_unary/models/"


# @pytest.mark.xfail(
#     reason="tenstorrent/pybuda#1"
# )
def test_eltwise_unary(
    un_train,
    un_recompute,
    un_op,
    un_model,
    un_shape,
    un_kwargs
):

    print("\n")
    print(f"un_train --> {un_train}")
    print(f"un_recompute --> {un_recompute}")
    print(f"un_op --> {un_op}")
    print(f"un_model --> {un_model}")
    print(f"un_shape --> {un_shape}")
    print(f"un_kwargs --> {un_kwargs}")
    print("\n")

    if not un_train and un_recompute:
        pytest.skip("Inference and recompute is the same as just inference.")
    
    assert type(un_train) in [bool, str], "Type of training parameter must be boolean or string"
    if type(un_train) == str:
        training = True if un_train == 'True' else False
    else:
        training = un_train
    assert type(un_recompute) in [bool, str], "Type of recompute parameter must be boolean or string"
    if type(un_recompute) == str:
        recompute = True if un_recompute == 'True' else False
    else:
        recompute = un_recompute
    operation = un_op
    model = un_model
    shape = eval(un_shape) if type(un_shape) == str else un_shape

    kwargs = un_kwargs
    pcc = 0.99
    if operation == "LeakyRelu":
        if un_train:
            pcc = 0.95
    

    print("\n")
    print(f"Training --> {training}")
    print(f"Recompute --> {recompute}")
    print(f"Operation --> {operation}")
    print(f"Model --> {model}")
    print(f"Shape --> {shape}")
    print(f"Kwargs --> {kwargs}")
    print("\n")

    architecture = f'models.{model}.BudaElementWiseUnaryTest(operator=pybuda.op.{operation}, opname="{operation}", shape={shape}'
    for k, v in kwargs.items():
        architecture = f'{architecture}, {k}={v}'
    architecture = f'{architecture})'

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
        verify_cfg=VerifyConfig(pcc=pcc)
    )