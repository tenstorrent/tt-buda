# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import tensorflow as tf

from pybuda import (
    TFModule,
    VerifyConfig,
    CompileDepth,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

input_shapes = [(1, 32, 32, 1)]

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
def test_mnist_tensorflow(test_kind, test_device, input_shape):
    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported
    
    class MNIST(tf.keras.Model):
        def __init__(self):
            super().__init__()

            self.conv_padding = tf.keras.layers.ZeroPadding2D(5//2)
            self.conv1 = tf.keras.layers.Conv2D(32, 5, padding="valid")
            self.maxpool_padding = tf.keras.layers.ZeroPadding2D(1)
            self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=3, padding="valid")
            self.conv2 = tf.keras.layers.Conv2D(64, 5, padding="valid")
            self.conv2_drop = tf.keras.layers.SpatialDropout2D(0.5)
            self.fc1 = tf.keras.layers.Dense(320, activation="relu")
            self.drop = tf.keras.layers.Dropout(0.5)
            self.fc2 = tf.keras.layers.Dense(10)

        def call(self, x):
            x = self.conv_padding(x)
            x = self.conv1(x)
            x = self.maxpool_padding(x)
            x = self.max_pool(x)
            x = tf.keras.activations.relu(x)
            x = self.conv_padding(x)
            x = self.conv2(x)
            x = self.conv2_drop(x)
            x = self.maxpool_padding(x)
            x = self.max_pool(x)
            x = tf.keras.activations.relu(x)
            x = tf.reshape(x, (-1, 1024))
            x = self.fc1(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = tf.math.softmax(x)
            return tf.math.log(x)

    model = MNIST()
    mod = TFModule("mnist", model)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
