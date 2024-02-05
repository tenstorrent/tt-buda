# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from transformers import TFConvNextModel, ConvNextConfig

from pybuda import (
    PyTorchModule,
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.tvm.utils import evaluate_framework_vs_pybuda

import tensorflow as tf

from pybuda.verify.backend import verify_module

def test_tvm_convnext(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    # tenstorrent/pybuda#842
    compiler_cfg.compile_depth = (
        CompileDepth.BUDA_GRAPH_PRE_PLACER
    )

    cfg = ConvNextConfig.from_pretrained("facebook/convnext-tiny-224")
    framework_model = TFConvNextModel(cfg).convnext.encoder.stages[0].layers[0].dwconv
    
    
    module = TFModule("convnext_full_model_tiny_tf", framework_model)
    input_shape = (1, 384, 96, 96)
    x = tf.random.normal(input_shape)
    framework_model(x)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
