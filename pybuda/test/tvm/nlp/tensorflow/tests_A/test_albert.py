# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import numpy as np
import tensorflow as tf
from transformers import AlbertConfig, TFAlbertModel
from transformers.models.albert.modeling_tf_albert import TFAlbertAttention, TFAlbertLayer

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
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


model_config_v1 = {
  "_name_or_path": "albert-base-v1",
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torchscript": True,
  "transformers_version": "4.12.2",
  "type_vocab_size": 2,
  "vocab_size": 30000
}


def test_albert_v1(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    
    class TF_AlbertAttention(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFAlbertAttention(config)

        def call(self, hidden_states):
            return self.layer(hidden_states, None, None, None)

    config = AlbertConfig(**model_config_v1)
    
    model = TF_AlbertAttention(config)
    mod = TFModule(
        "albert_attention_tf",
        model,
    )
    input_shape = (1, 768, 768)

    relative_atol = 0.1
    if test_kind == TestKind.INFERENCE and test_device.devtype == BackendType.Silicon:
        relative_atol = 0.3

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            relative_atol=relative_atol,
            waive_gradient_errors={"tf__albert_attention/tf_albert_attention/key/bias:0",
                "tf__albert_attention_1/tf_albert_attention_1/key/bias:0"},
        )
    )


model_config_v2 = {
  "_name_or_path": "albert-base-v2",
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torchscript": True,
  "transformers_version": "4.12.2",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

def test_albert_v2(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    
    class TF_AlbertAttention(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFAlbertAttention(config)

        def call(self, hidden_states):
            return self.layer(hidden_states, None, None, None)

    config = AlbertConfig(**model_config_v2)
    
    model = TF_AlbertAttention(config)
    mod = TFModule(
        "albert_attention_tf",
        model,
    )
    input_shape = (1, 768, 768)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"tf__albert_attention/tf_albert_attention/key/bias:0",
                "tf__albert_attention_3/tf_albert_attention_3/key/bias:0"},
        )
    )
