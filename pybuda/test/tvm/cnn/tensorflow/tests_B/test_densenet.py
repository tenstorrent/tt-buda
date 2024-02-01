# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
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
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda

input_shapes = [(1, 224, 224, 3)]

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
def test_densenet_tf(training, input_shape):
    recompute = True

    if training:
        pytest.skip()  # Backward is currently unsupported

    compile_depth = CompileDepth.PRE_LOWERING_PASS

    model = tf.keras.applications.densenet.DenseNet121(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    mod = TFModule("densenet121_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = tf.random.uniform(input_shape)

    ret = pybuda_compile(
        tt0,
        "densenet121_tf",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth,
            balancer_policy="CNN"
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(model, ret, act1)