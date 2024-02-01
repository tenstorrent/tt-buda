# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# AutoEncoder basic bring-up tests of tracing functionality
#
import pytest

from keras import Model
from keras import layers
import tensorflow as tf

from pybuda import (
    TFModule,
    VerifyConfig,
)
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module
from pybuda.config import CompileDepth, _get_global_compiler_config


def test_conv_autoencoder(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class ConvAE(Model):
        def __init__(self):
            super(ConvAE, self).__init__()
            self.encoder = tf.keras.Sequential(
                [
                    layers.Conv2D(
                        filters=16,
                        kernel_size=3,
                        strides=2,
                        activation="relu",
                        padding="valid",
                    ),
                    layers.Conv2D(
                        filters=8,
                        kernel_size=3,
                        strides=2,
                        activation="relu",
                        padding="valid",
                    ),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    layers.Conv2DTranspose(
                        filters=8,
                        kernel_size=3,
                        strides=2,
                        activation="relu",
                        padding="valid",
                    ),
                    layers.Conv2DTranspose(
                        filters=16,
                        kernel_size=3,
                        strides=2,
                        activation="relu",
                        padding="valid",
                    ),
                ]
            )

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)

            return decoded

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        # Column dimension must be divisible by tile size
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    framework_model = ConvAE()
    module = TFModule(
        "tf_conv_autoencoder",
        framework_model,
    )

    input_shape = (1, 28, 28, 3)

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
