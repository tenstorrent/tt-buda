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
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.tvm.utils import evaluate_framework_vs_pybuda

import tensorflow as tf

@pytest.mark.skip(reason="Takes too long to run on postcommit")
def test_tvm_nasnet_mobile_tf(training=False):
    recompute = True    # Always run with recompute in post-commit CI. Nightly tests both

    model = tf.keras.applications.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )

    mod = TFModule("nasnet_mobile_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = tf.random.uniform((1, 224, 224, 3))

    ret = pybuda_compile(
        tt0,
        "nasnet_mobile_tf",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.PRE_LOWERING_PASS,
            balancer_policy="CNN"
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, ret, act1)