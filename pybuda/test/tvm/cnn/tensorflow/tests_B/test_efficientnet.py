# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import tensorflow as tf

from pybuda import (
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    CompileDepth,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import _get_global_compiler_config
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module
import pybuda


input_shapes = [(1, 1, 32, 16)]

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
def test_transpose_batch_dim(training, input_shape):
    recompute = True    # Always run with recompute in post-commit CI. Nightly tests both
    if training:
        pytest.skip()  # Backward is currently unsupported

    class Transpose(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return tf.transpose(x, [3, 1, 2, 0])

    model = Transpose()

    mod = TFModule("transpose_batch_dim_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = tf.random.uniform(input_shape)
    x = model(act1)

    ret = pybuda_compile(
        tt0,
        "transpose_batch_dim_tf",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            enable_tvm_constant_prop=True,
            balancer_policy="CNN",
            compile_depth = CompileDepth.PRE_LOWERING_PASS, # Unsupported HW ops
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(model, ret, act1)


def test_efficientnet_layer(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    compiler_cfg.balancer_policy = "CNN"

    blocks_args = [{
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
    }]

    input_shape = (1, 32, 112, 112)
    model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape[1:], blocks_args=blocks_args, weights=None)

    mod = TFModule("efficientnet_b0_layer_tf", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
