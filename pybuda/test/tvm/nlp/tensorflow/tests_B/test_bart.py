# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from base64 import encode
from distutils.config import PyPIRCCommand
import pytest

import tensorflow as tf
import torch

from transformers import BartConfig, TFBartModel, BartTokenizer
from transformers.models.bart.modeling_tf_bart import _make_causal_mask, shift_tokens_right
from test.backend.models.test_bert import get_relaxed_atol_pcc

import pybuda
from pybuda import (
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    CPUDevice,
    TTDevice,
    Tensor,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model
from loguru import logger


from pybuda.op.eval.common import compare_tensor_to_golden

def test_bart_decoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 768, 768)

    pretrained_name = "facebook/bart-base"
    config = download_model(BartConfig.from_pretrained, pretrained_name, torchscript=True)
 
    model = download_model(TFBartModel.from_pretrained, pretrained_name, config=config)
    mod = TFModule("bart_decoder_block", model.get_decoder().layers[0])

    
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_bart_encoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 768, 768)

    pretrained_name = "facebook/bart-base"

    class TF_BartEncoderBlock(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def call(self, x):
            return self.model(x, None, None)

    config = download_model(BartConfig.from_pretrained, pretrained_name, torchscript=True)
 
    model = download_model(TFBartModel.from_pretrained, pretrained_name, config=config).get_encoder().layers[0]
    model = TF_BartEncoderBlock(model)
    mod = TFModule("bart_encoder_block", model)

    atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)


    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"k_proj/bias"},
            relative_atol=atol,
            pcc=pcc
        )
    )


class EmbWrapper(tf.keras.Model):
    def __init__(self, shared_embeddings, embeddings):
        super().__init__()
        self.shared_embed = shared_embeddings
        self.encoder_decoder_embeddings = embeddings

    def call(self, input_ids):
        inputs_embeds = self.shared_embed(input_ids)
        embed_pos = self.encoder_decoder_embeddings(input_ids.shape)
        return inputs_embeds + embed_pos

class BlocksWrapper(tf.keras.Model):
    def __init__(self, module, attention_mask=None, layer_head_mask=None):
        super().__init__()
        self.encoder_decoder = module
        self.layernorm_embedding = self.encoder_decoder.layernorm_embedding
        self.attention_mask = attention_mask
        self.layer_head_mask = layer_head_mask

    def call(self, hidden_states):
        hidden_states = self.layernorm_embedding(hidden_states)
        for block in self.encoder_decoder.layers:
            hidden_states = block(
                hidden_states,
                self.attention_mask,
                self.layer_head_mask
            )[0]
        
        return hidden_states
@pytest.mark.skip(reason="Tested with fallback")
def test_bart_encoder_pipeline(test_device):

    pretrained_name = "facebook/bart-base"
    config = download_model(BartConfig.from_pretrained, pretrained_name, torchscript=True)

    model = download_model(TFBartModel.from_pretrained, pretrained_name, config=config)

    encoder_embeddings = EmbWrapper(model.model.shared, model.get_encoder().embed_positions)

    encoder_blocks = BlocksWrapper(model.get_encoder())

    cpu0 = CPUDevice("cpu0", module=TFModule("encoder_embeddings", encoder_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=TFModule("encoder_blocks", encoder_blocks))

    seq_len = 768
    input_ids = tf.random.uniform((1, seq_len), maxval=config.vocab_size, dtype=tf.int32)
    
    cpu0.push_to_inputs(input_ids)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(verify_last=False))
    outputs = output_q.get()
    
    tf_outputs = model.get_encoder()(input_ids)
    assert compare_tensor_to_golden("bart_encoder", tf_outputs[0], outputs[0].value(), is_buda=True)

@pytest.mark.skip(reason="Tested with fallback")
def test_bart_decoder_pipeline(test_device):

    pretrained_name = "facebook/bart-base"
    config = download_model(BartConfig.from_pretrained, pretrained_name, torchscript=False)

    model = download_model(TFBartModel.from_pretrained, pretrained_name, config=config)

    decoder_embeddings = EmbWrapper(model.model.shared, model.get_decoder().embed_positions)

    seq_len = 768
    input_ids = tf.convert_to_tensor(tf.random.uniform((1, seq_len), maxval=config.vocab_size, dtype=tf.int32))
    input_ids = shift_tokens_right(
        input_ids, config.pad_token_id, config.decoder_start_token_id
    )
    
    attn_mask = _make_causal_mask(input_ids.shape)

    decoder_blocks = BlocksWrapper(model.get_decoder(), attention_mask=attn_mask)

    cpu0 = CPUDevice("cpu0", module=TFModule("decoder_embeddings", decoder_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=TFModule("decoder_blocks", decoder_blocks))

    tf_outputs = model.get_decoder()(input_ids)
    cpu0.push_to_inputs(input_ids)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(verify_last=True))
    outputs = output_q.get()
 
    assert compare_tensor_to_golden("bart_decoder", tf_outputs[0], outputs[0].value(), is_buda=True)


@pytest.mark.parametrize("size", ["base", "large"])
def test_bart_tf(test_kind, test_device, size):
    if size == "large":
        pytest.skip() # too slow for post commit
        
    if test_kind.is_training():
        pytest.skip()

    model_name = f"facebook/bart-{size}"
    config = download_model(BartConfig.from_pretrained, model_name, torchscript=False)
    model = download_model(TFBartModel.from_pretrained, model_name, config=config)

    #pre-process 
    tokenizer = download_model(BartTokenizer.from_pretrained, model_name, pad_to_max_length=True)
    example = ["Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."]
    inputs_dict = tokenizer(example, return_tensors='tf', max_length=96, padding="max_length")    
    decoder_input_ids = shift_tokens_right(inputs_dict["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id)
    inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], decoder_input_ids]
    input_shapes = tuple([tuple(i.shape) for i in inputs])  

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("where")

    mod = TFModule("bart_tf", model)
 
    pcc = 0.95 if test_device.devtype == BackendType.Silicon else 0.99  
    verify_module(
        mod,
        ((1,96), (1,96), (1,96),),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,  
            pcc=pcc,
        ), 
    ) 

