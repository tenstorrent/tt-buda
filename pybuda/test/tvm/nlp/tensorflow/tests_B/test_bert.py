# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import numpy as np
import tensorflow as tf
import torch
from transformers import BertConfig
from transformers.models.bert.modeling_tf_bert import TFBertLayer, TFBertModel, TFBertMainLayer, TFBertForQuestionAnswering, TFBertAttention
from pybuda import BackendDevice

from pybuda import (
    CPUDevice,
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from test.backend.models.test_bert import get_relaxed_atol_pcc
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module, verify_module_pipeline
from pybuda.verify.config import TestKind
from test.utils import download_model
from test.backend.models.test_bert import get_relaxed_atol_pcc

model_config = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "pad_token_id": 0,
    "type_vocab_size": 2,
    "vocab_size": 30522,
}

def test_bert(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()
        
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Data mismatch on output")
        
    class TF_BertLayer(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFBertLayer(config)

        def call(self, hidden_states):
            return self.layer(hidden_states, None, None, None, None, None, False)

    config = BertConfig(**model_config)
    model = TF_BertLayer(config=config)
    mod = TFModule("BertLayer", model)

    input_shape = (1, 64, 128)
    hidden_states = tf.convert_to_tensor(
        np.random.rand(*input_shape).astype(np.float32)
    )

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            relative_atol=relative_atol, 
            pcc=pcc,
            waive_gradient_errors={"attention/self/key/bias:0"},
        )
    )


def test_bert_pipeline(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()
        
    class EncoderWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.layers[0].encoder

        def call(self, hidden_states, attention_mask):
            head_mask = [None] * self.encoder.config.num_hidden_layers
            return self.encoder(hidden_states, attention_mask, head_mask, None, None, None, False, False, False, False)

    class EmbeddingsWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.embeddings = model.layers[0].embeddings

        def call(self, input_ids, attention_mask):
            embedding_output = self.embeddings(input_ids)
            return embedding_output, attention_mask


    model = download_model(TFBertModel.from_pretrained, "prajjwal1/bert-tiny", from_pt=True)
    embeddings = TFModule("embeddings", EmbeddingsWrapper(model))
    encoder = TFModule("encoders", EncoderWrapper(model))

    microbatch_size = 1
    seq_len = 64
    vocab_size = model.layers[0].embeddings.config.vocab_size
    input_ids = tf.Variable(tf.random.uniform((microbatch_size, seq_len), maxval=vocab_size, dtype=tf.dtypes.int32), trainable=False)

    attention_mask = tf.fill(dims=(2, microbatch_size, seq_len), value=1)

    extended_attention_mask = tf.reshape(
        attention_mask, (microbatch_size, model.config.num_attention_heads, 1, seq_len)
    )
    extended_attention_mask = tf.cast(extended_attention_mask, dtype=tf.dtypes.float32)
    one_cst = tf.constant(1.0)
    ten_thousand_cst = tf.constant(-10000.0)
    extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

    waive_gradients = {"attention/self/key/bias:0"}
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module_pipeline([embeddings, encoder],
            [(microbatch_size, seq_len), (microbatch_size, 1, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol, pcc=pcc, 
                waive_gradient_errors=waive_gradients),
            inputs=[(input_ids, extended_attention_mask), ],
            input_params=[{"requires_grad": False}, {"requires_grad": False}],
            device_types=["CPUDevice", "TTDevice"],
    )


def test_bert_tf_fallback(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    input_shape = (1, 128)
    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny")
    model = TFBertMainLayer(config, add_pooling_layer=False)
    submodel = model
    mod = TFModule("tf_bert", submodel)

    compiler_cfg = _get_global_compiler_config() 

    atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"key/bias"},
            relative_atol=atol,
            pcc=pcc
        ),
        input_params=[{"requires_grad": False, "data_format": tf.int32}],
    )


def test_bert_qa_tf_fallback(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    input_shape = (1, 128)
    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny")
    model = TFBertForQuestionAnswering(config)
    submodel = model
    mod = TFModule("tf_bert_for_qa_const_prop", submodel)

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask = {"qa_outputs"}

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={'key/bias'},
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )


def test_bert_self_attention(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING: 
        pytest.skip()

    # TODO: Remove
    if test_kind.is_training():
        pytest.skip()

    class SelfAttention(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFBertAttention(config)

        def call(self, hidden_states):
            return self.layer(hidden_states, None, None, None, None, None, False)

    # Initialize module
    config = BertConfig(**model_config)
    framework_module = SelfAttention(config=config)
    pybuda_module = TFModule("bert_self_attention_tf", framework_module)
    input_shape = (1, 128, 128)

    # Run module
    # hidden_states = tf.convert_to_tensor(np.random.rand(*input_shape).astype(np.float32))
    # res = framework_module(hidden_states, None, None, None, None, None, None)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
