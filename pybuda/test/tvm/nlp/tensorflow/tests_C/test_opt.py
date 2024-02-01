# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest
from typing import Optional, Tuple
import torch
import tensorflow as tf
from transformers import OPTConfig
from transformers.models.opt.modeling_tf_opt import TFOPTDecoderLayer, TFOPTModel
from pybuda import (
    TFModule,
    TTDevice,
    CPUDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)


from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module, verify_module_pipeline
from pybuda.verify.config import TestKind

from loguru import logger
import pybuda
from pybuda.op.eval.common import compare_tensor_to_golden
from test.utils import download_model


class EmbWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.decoder.embed_tokens
        self.embed_positions = model.model.decoder.embed_positions

    def call(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        attention_mask = tf.ones(inputs_embeds.shape[:2], dtype=tf.float32)
        pos_embeds = self.embed_positions(attention_mask, 0)

        return inputs_embeds, attention_mask, pos_embeds
class DecoderWrapper(tf.keras.Model):
    def __init__(self, model, config):
        super().__init__()
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = model.model.decoder.final_layer_norm
        else:
            self.final_layer_norm = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = model.model.decoder.project_out
            self.project_in = model.model.decoder.project_in
        else:
            self.project_in = None
            self.project_out = None
        self.decoders = [x for x in model.model.decoder.layers] # reduce compile time

        self._prepare_decoder_attention_mask = model.model.decoder._prepare_decoder_attention_mask


    def call(self, inputs_embeds, attention_mask, pos_embeds):
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, inputs_embeds.shape[:2], 0)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        for idx, decoder_layer in enumerate(self.decoders):
            hidden_states, layer_self_attn, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask= None,
                past_key_value=None,
            )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


def test_tf_opt_pipeline(test_device, test_kind):
    pytest.skip() # tested with fallback below

    # Broadcast backwards not implemented
    if test_kind.is_training():
        pytest.skip()

    from transformers import TFOPTModel
    tf.keras.backend.clear_session()
    model = download_model(TFOPTModel.from_pretrained, "facebook/opt-350m")
    config = model.config

    # WORKAROUND FOR RELU
    for layer in model.model.decoder.layers:
        layer.activation_fn = tf.keras.activations.gelu

    # Reduce compile time
    model.model.decoder.layers =  model.model.decoder.layers[0:1]
    opt_embeddings = EmbWrapper(model)
    opt_decoder = DecoderWrapper(model, config)

    embeddings = TFModule("embeddings", opt_embeddings)
    decoder = TFModule("decoder", opt_decoder)

    input_ids = tf.Variable(tf.random.uniform(shape=(1, 32), minval=0, maxval=config.vocab_size, dtype=tf.int32), trainable=False)

    verify_module_pipeline([embeddings, decoder],
            [(1, 32)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch,
            accumulation_steps=1, relative_atol=0.3,),
            inputs=[(input_ids, ), ],
            input_params=[{"requires_grad": False}],
            device_types=["CPUDevice", "TTDevice"],
    )



def test_opt_decoder_tf(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()


    configuration = OPTConfig()
    model = TFOPTDecoderLayer(configuration)

    mod = TFModule("OPT_decoder_layer", model)

    input_shape = (1, 32, 768)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_opt_fallback(test_kind, test_device):
    if test_kind.is_training(): # only run recompute test in post-commit
        pytest.skip()
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop=True 

    configuration = OPTConfig()

    model = TFOPTModel(configuration)

    mod = TFModule("OPT_tf", model)

    input_shape = (1, 768)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        input_params=[
            {"requires_grad": False, "data_format": tf.int32}, 
        ],
    )
