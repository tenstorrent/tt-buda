# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

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
from pybuda.config import CompileDepth
from pybuda.config import _get_global_compiler_config
from pybuda.verify.backend import verify_module

import tensorflow as tf
import tensorflow_hub as hub

def test_tvm_regnety002_tf(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    model = tf.keras.applications.regnet.RegNetY002(
        model_name='regnety002',
        include_top=True,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )

    mod = TFModule("regnety002", model)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    verify_module(
        mod,
        ((1, 224, 224, 3),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
