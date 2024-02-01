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
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
import tensorflow as tf
from pybuda.config import CompileDepth, _get_global_compiler_config


def test_tvm_resnet_tf(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    
    mod = TFModule("resnet50_tf", model)

    verify_module(
        mod,
        ((1, 224, 224, 3),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_resnet_rs_tf(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    model = tf.keras.applications.resnet_rs.ResNetRS50(
        include_top=True,
        weights='imagenet',
        classes=1000,
        input_shape=None,
        input_tensor=None,
        pooling=None,
        classifier_activation='softmax',
        include_preprocessing=True
    )
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    mod = TFModule("resnet50_rs_tf", model)

    verify_module(
        mod,
        ((1, 224, 224, 3),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
