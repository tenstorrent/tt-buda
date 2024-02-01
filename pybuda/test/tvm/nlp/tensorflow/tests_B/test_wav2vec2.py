# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Wav2Vec2 basic bring-up tests of tracing functionality
#
import pytest

import tensorflow as tf
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_tf_wav2vec2 import (
    TFWav2Vec2Model,
    TFWav2Vec2FeatureEncoder,
    TFWav2Vec2FeatureProjection,
    TFWav2Vec2WeightNormConv1D,
    TFWav2Vec2Encoder,
)

from pybuda import (
    TFModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_wav2vec2_full_model(test_kind, test_device):
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        # Unsupported HW ops
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    else:
        # Unsupported backward pass for concatenate op
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    config = Wav2Vec2Config()
    framework_module = TFWav2Vec2Model(config)

    module = TFModule(
        "wav2vec2_feature_encoder",
        framework_module,
    )

    input_shape = (1, 512)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_wav2vec2_feature_encoder(test_kind, test_device):
    pytest.skip()  # Tested in full model, useful for debugging.
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    class TFWav2Vec2_FeatureEncoder(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFWav2Vec2FeatureEncoder(config)

        def call(self, input_values):
            return self.layer(
                input_values,
            )

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        # Unsupported HW ops
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        # TODO: Tensor mismatch on bw_in0_gelu_197_multiply_1 from layernorm_196
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    config = Wav2Vec2Config()
    framework_module = TFWav2Vec2_FeatureEncoder(config)
    module = TFModule(
        "wav2vec2_feature_encoder",
        framework_module,
    )

    input_shape = (1, 512)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_wav2vec2_feature_projection(test_kind, test_device):
    pytest.skip()  # Tested in full model, useful for debugging.
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    class TFWav2Vec2_FeatureProjection(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFWav2Vec2FeatureProjection(config, name="feature_projection")

        def call(self, input_values):
            return self.layer(
                input_values,
            )

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        # Segmentation fault on balancer - access params through backend api
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        # Segmentation fault on balancer - access params through backend api
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    config = Wav2Vec2Config()
    framework_module = TFWav2Vec2_FeatureProjection(config)
    module = TFModule(
        "wav2vec2_feature_projection",
        framework_module,
    )

    input_shape = (1, 512)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_wav2vec2_conv1d_with_norm(test_kind, test_device):
    pytest.skip()  # Tested in full model, useful for debugging.
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    class TFWav2Vec2_WeightNormConv1D(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFWav2Vec2WeightNormConv1D(
                filters=config.hidden_size,
                kernel_size=config.num_conv_pos_embeddings,
                groups=config.num_conv_pos_embedding_groups,
                explicit_padding=config.num_conv_pos_embeddings // 2,
                name="conv",
            )

        def call(self, input_values):
            return self.layer(
                input_values,
            )

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL
        pytest.skip()

    config = Wav2Vec2Config()
    config.num_hidden_layers = 1
    framework_module = TFWav2Vec2_WeightNormConv1D(config)
    module = TFModule(
        "wav2vec2_conv1d_with_norm",
        framework_module,
    )

    input_shape = (1, 1, 768)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_wav2vec2_encoder(test_kind, test_device):
    pytest.skip()  # Tested in full model, useful for debugging.
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    class TFWav2Vec2_Encoder(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFWav2Vec2Encoder(config, name="encoder")

        def call(self, input_values):
            return self.layer(
                input_values,
            )

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL

    config = Wav2Vec2Config()
    config.num_hidden_layers = 1
    framework_module = TFWav2Vec2_Encoder(config)
    module = TFModule(
        "wav2vec2_encoder",
        framework_module,
    )

    input_shape = (1, 1, 768)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
