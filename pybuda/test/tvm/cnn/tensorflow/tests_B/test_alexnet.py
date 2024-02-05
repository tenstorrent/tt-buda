# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# AlexNet basic bring-up tests of tracing functionality
#
import pytest

import keras
from keras import Model
import tensorflow as tf

from pybuda import (
    TFModule,
    VerifyConfig,
)
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module
from pybuda.config import CompileDepth, _get_global_compiler_config


def test_alexnet(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # Training is currently unsupported for CNNs
    if test_kind.is_training():
        pytest.skip()

    class AlexNet(Model):
        def __init__(self):
            super(AlexNet, self).__init__()

            self.features = keras.models.Sequential(
                [
                    keras.layers.Conv2D(
                        filters=64,
                        kernel_size=11,
                        strides=4,
                        padding="same",
                        activation="relu",
                        # input_shape=(224, 224, 3),
                        input_shape=(128, 128, 3),
                    ),
                    keras.layers.MaxPool2D(
                        pool_size=3,
                        strides=2,
                    ),
                    keras.layers.Conv2D(
                        filters=192,
                        kernel_size=5,
                        strides=1,
                        padding="same",
                        activation="relu",
                    ),
                    keras.layers.MaxPool2D(
                        pool_size=3,
                        strides=2,
                    ),
                    keras.layers.Conv2D(
                        filters=384,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="relu",
                    ),
                    keras.layers.Conv2D(
                        filters=256,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="relu",
                    ),
                    keras.layers.Conv2D(
                        filters=256,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="relu",
                    ),
                    keras.layers.MaxPool2D(
                        pool_size=3,
                        strides=2,
                    ),
                ]
            )
            self.adaptive_avg_pool = keras.layers.AveragePooling2D(
                pool_size=1,
                strides=1,
                padding="valid",
            )
            self.flatten = keras.layers.Flatten()
            self.classifier = keras.models.Sequential(
                [
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(4096, activation="relu"),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(4096, activation="relu"),
                    keras.layers.Dense(1000, activation="softmax"),
                ]
            )

        def call(self, x):
            x = self.features(x)
            x = self.adaptive_avg_pool(x)
            x = self.flatten(x)
            x = self.classifier(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    framework_model = AlexNet()
    module = TFModule(
        "tf_alexnet",
        framework_model,
    )

    # input_shape = (1, 224, 224, 3)
    input_shape = (1, 128, 128, 3)

    # Run model
    # act = tf.random.uniform(input_shape)
    # out = framework_model(act)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
