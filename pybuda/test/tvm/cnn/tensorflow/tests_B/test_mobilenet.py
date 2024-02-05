# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import tensorflow as tf


from pybuda import (
    TTDevice,
    pybuda_compile,
    VerifyConfig,
    TFModule,
    CompilerConfig,
    optimizers,
    CompileDepth,
    BackendType,
    pybuda_reset,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module

def test_mobilenetv1_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    compiler_cfg.balancer_policy = "CNN"

    input_shape = (1, 224, 224, 3)

    act1 = tf.random.uniform(input_shape)

    model = tf.keras.applications.MobileNet (
        input_shape=input_shape[1:]
    )
    mod = TFModule("mobilenetv1_tf", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

def test_mobilenetv2_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    compiler_cfg.balancer_policy = "CNN"

    input_shape = (1, 224, 224, 3)

    model = tf.keras.applications.MobileNetV2 (
        input_shape=input_shape[1:]
    )
    mod = TFModule("mobilenetv2_tf", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

def test_mobilenetv3_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    compiler_cfg.balancer_policy = "CNN"

    input_shape = (1, 224, 224, 3)

    model = tf.keras.applications.MobileNetV3Small(
        input_shape=None,
        alpha=1.0,
        minimalistic=False,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation='softmax',
        include_preprocessing=True
    )
    mod = TFModule("mobilenetv3_tf", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )