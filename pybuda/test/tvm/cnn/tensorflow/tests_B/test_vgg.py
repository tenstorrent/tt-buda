# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pybuda.config import CompileDepth
import pytest

import tensorflow as tf

from pybuda import (
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

def test_tvm_vgg16_tf(training):
    recompute = True    # Always run with recompute in post-commit CI. Nightly tests both
    if training:
        pytest.skip()

    compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    mod = TFModule("vgg16_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = tf.random.uniform((1, 224, 224, 3))

    ret = pybuda_compile(
        tt0,
        "vgg16_tf",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth,
            balancer_policy="CNN"
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
            pcc=0.97,
        ),
    )
    evaluate_framework_vs_pybuda(model, ret, act1)

if __name__ == "__main__":
    test_tvm_vgg16_tf(False, False)
