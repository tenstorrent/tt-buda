# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os
from base64 import encode
from distutils.config import PyPIRCCommand
import pytest

import torch
from transformers import BartConfig, BartModel, BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right, BartAttention
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from test.backend.models.test_bert import get_relaxed_atol_pcc

import pybuda
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    CPUDevice,
    TTDevice,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

from loguru import logger


from pybuda.op.eval.common import compare_tensor_to_golden

def test_bart_decoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    input_shape = (1, 768, 768)

    pretrained_name = "facebook/bart-base"
    config = BartConfig.from_pretrained(pretrained_name, torchscript=True)
 
    model = BartModel(config)
    pretrained_model = BartModel.from_pretrained(pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())

    mod = PyTorchModule("bart_decoder_block", model.decoder.layers[0])

    
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
    attn_mask_shape = (1, 1, 768, 768)
    layer_head_mask_shape = (12,)

    pretrained_name = "facebook/bart-base"
    config = BartConfig.from_pretrained(pretrained_name, torchscript=True)

    model = BartModel(config)
    pretrained_model = BartModel.from_pretrained(pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.layer = encoder
            self.attn_mask = torch.randn(attn_mask_shape)
            self.layer_head_mask = torch.randn(layer_head_mask_shape)

        def forward(self, hidden_states):
            return self.layer(hidden_states, self.attn_mask, self.layer_head_mask)

    mod = PyTorchModule("bart_encoder_block", EncoderWrapper(model.encoder.layers[0]))

    atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"k_proj.bias"},
            relative_atol=atol,
            pcc=pcc
        )
    )


class EmbWrapper(torch.nn.Module):
    def __init__(self, shared_embeddings, embeddings):
        super().__init__()
        self.shared_embed = shared_embeddings
        self.encoder_decoder_embeddings = embeddings

    def forward(self, input_ids):
        inputs_embeds = self.shared_embed(input_ids)
        embed_pos = self.encoder_decoder_embeddings(input_ids.shape)
        return inputs_embeds + embed_pos

class BlocksWrapper(torch.nn.Module):
    def __init__(self, module, attention_mask=None, layer_head_mask=None):
        super().__init__()
        self.encoder_decoder = module
        self.layernorm_embedding = self.encoder_decoder.layernorm_embedding
        self.attention_mask = attention_mask
        self.layer_head_mask = layer_head_mask

    def forward(self, hidden_states):
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
    config = BartConfig.from_pretrained(pretrained_name, torchscript=True)

    model = BartModel(config)
    pretrained_model = BartModel.from_pretrained(pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())
    model.eval()

    encoder_embeddings = EmbWrapper(model.shared, model.encoder.embed_positions)

    encoder_blocks = BlocksWrapper(model.encoder)

    cpu0 = CPUDevice("cpu0", module=PyTorchModule("encoder_embeddings", encoder_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("encoder_blocks", encoder_blocks))

    seq_len = 768
    input_ids = torch.randint(config.vocab_size, (1, seq_len))
    
    cpu0.push_to_inputs(input_ids)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(verify_last=False))
    outputs = output_q.get()
    
    torch_outputs = model.encoder(input_ids)
    assert compare_tensor_to_golden("bart_encoder", torch_outputs[0], outputs[0].value(), is_buda=True)

@pytest.mark.skip(reason="Tested with fallback")
def test_bart_decoder_pipeline(test_device):

    pretrained_name = "facebook/bart-base"
    config = BartConfig.from_pretrained(pretrained_name, torchscript=True)

    model = BartModel(config)
    pretrained_model = BartModel.from_pretrained(pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())
    model.eval()

    decoder_embeddings = EmbWrapper(model.shared, model.decoder.embed_positions)

    seq_len = 768
    input_ids = torch.randint(config.vocab_size, (1, seq_len))
    input_ids = shift_tokens_right(
        input_ids, config.pad_token_id, config.decoder_start_token_id
    )
    
    attn_mask = _create_4d_causal_attention_mask(input_ids.shape, torch.float32)
    decoder_blocks = BlocksWrapper(model.decoder, attention_mask=attn_mask)

    cpu0 = CPUDevice("cpu0", module=PyTorchModule("decoder_embeddings", decoder_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("decoder_blocks", decoder_blocks))

    cpu0.push_to_inputs(input_ids)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig())
    outputs = output_q.get()

    torch_outputs = model.decoder(input_ids)
    assert compare_tensor_to_golden("bart_decoder", torch_outputs[0], outputs[0].value(), is_buda=True)


@pytest.mark.parametrize("size", ["base", "large"])
def test_pt_bart(test_kind, test_device, size):
    if size == "large":
        pytest.skip() # too slow for post commit
        
    if test_kind.is_training():
        pytest.skip()
        
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, attn_mask):
            super().__init__()
            self.model = model
            self.attention_mask = attn_mask

        def forward(self, input_ids, decoder_input_ids):
            return self.model(input_ids=input_ids, attention_mask=self.attention_mask, decoder_input_ids=decoder_input_ids)

    model_name = f"facebook/bart-{size}"
    model = BartModel.from_pretrained(model_name, torchscript=True)

    #pre-process 
    tokenizer = BartTokenizer.from_pretrained(model_name, pad_to_max_length=True)
    example = ["Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."]
    inputs_dict = tokenizer(example, return_tensors='pt', max_length=96, padding="max_length")    
    decoder_input_ids = shift_tokens_right(inputs_dict["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id)
    inputs = [inputs_dict["input_ids"], decoder_input_ids]

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("where")

    model = ModelWrapper(model, inputs_dict["attention_mask"])
    mod = PyTorchModule("bart_full", model)

    #relative_atol = 0.2 if size == "large" else 0.1 
    pcc = 0.95 if test_device.devtype == BackendType.Silicon else 0.99  
    verify_module(
        mod,
        ((1,96), (1,96),),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,  
            pcc=pcc,
        ), 
    ) 

