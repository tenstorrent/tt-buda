# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
""" 
GPT2 tests on backend
"""

import math
import copy
import os
import torch
import pytest
import inspect

from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig
from pybuda import Tensor, DataFormat, BackendType

import pybuda
from pybuda import PyTorchModule
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from pybuda.config import CompilerConfig, _get_global_compiler_config

from loguru import logger

def test_pt_gpt2_block(test_kind, test_device):
    model = GPT2Model.from_pretrained("gpt2")
    block = PyTorchModule("gpt2_block_backend",model.h[0])

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask = {"attn.c_attn.weight", "attn.c_attn.bias"}

    if test_kind.is_training():
        pytest.skip()

    relative_atol = 0.3 if test_kind.is_training() else 0.1

    verify_module(block, [(1, 64, 768),],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol,),
            input_params=[{"requires_grad": False}],
    )

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


from transformers import pipeline, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
def test_pt_gpt2_fallback(test_kind, test_device):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prefix_text = "My name is Bert, and I am"

    text_generator_pt = pipeline("text-generation", model=model, tokenizer=tokenizer)
    torch.manual_seed(42)
    answer_pt = text_generator_pt(
        prefix_text, 
        max_length=10, 
        num_beams=4,
        num_return_sequences=4, 
        pad_token_id=50256, 
        no_repeat_ngram_size=2,
    )

    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer,)
    torch.manual_seed(42)
    answer = text_generator(
        prefix_text, 
        max_length=10,  
        num_beams=4, 
        num_return_sequences=4, 
        pad_token_id=50256, 
        no_repeat_ngram_size=2,
    )
    
    logger.info(f"TT Generated text: {answer}")
    logger.info(f"PT Generated text: {answer_pt}")

    # assert answer == answer_pt

class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gpt2 = model.transformer

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.gpt2.wte(input_ids)
        position_ids = torch.arange(len(input_ids[0])).unsqueeze(0)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        return hidden_states, extended_attention_mask

class BlocksWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gpt2 = model.transformer

    def forward(self, hidden_states, extended_attention_mask):
        for block in self.gpt2.h:
            hidden_states = block(
                hidden_states,
                attention_mask=extended_attention_mask
            )[0]
        hidden_states = self.gpt2.ln_f(hidden_states)
        return hidden_states

class LMHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)

from transformers import pipeline
@pytest.mark.skip(reason="Tested with fallback")
def test_pt_gpt2(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask = {"attn.c_attn.weight", "attn.c_attn.bias"}

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    embeddings = EmbWrapper(model)
    blocks = BlocksWrapper(model)
    lm_head = LMHeadWrapper(model)


    prefix_text = "My name is Bert, and I am"
    inputs = tokenizer(prefix_text, max_length=64, pad_to_max_length=True, truncation=True)
    input_ids_tt = torch.tensor(inputs["input_ids"]).int().unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"]).int().unsqueeze(0)

    last_prefix_token =  inputs["attention_mask"].index(0) - 1 
    tokens_to_generate = 20

    cpu0 = pybuda.CPUDevice("cpu0", module=PyTorchModule("gpt2_embeddings", embeddings))
    tt1 = pybuda.TTDevice("tt1", 
            devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("gpt2_blocks", blocks))
    cpu1 = pybuda.CPUDevice("cpu1", module=PyTorchModule("gpt2_lm_head", lm_head))


    for i in range(tokens_to_generate):
        cpu0.push_to_inputs((input_ids_tt, attention_mask))
        output_q = pybuda.run_inference()
        outputs = output_q.get()
        lm_head_out = outputs[0].value().detach()
        next_token = torch.argmax(lm_head_out, dim=-1)[0][last_prefix_token + i]
        next_token_index = last_prefix_token + i + 1
        input_ids_tt[0][next_token_index] = next_token
        attention_mask[0][next_token_index] = 1

    generated_text_tt = tokenizer.decode(input_ids_tt[0][:next_token_index].numpy().tolist())
    
    prefix_text = "My name is Bert, and I am"
    inputs = tokenizer(prefix_text, max_length=64, pad_to_max_length=True, truncation=True)
    input_ids_pt = torch.tensor(inputs["input_ids"]).int().unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"]).int().unsqueeze(0)

    for i in range(tokens_to_generate):
        embedding_output = embeddings(input_ids_pt, attention_mask)
        model_output = blocks(*embedding_output)
        lm_head_out = lm_head(model_output)
        next_token = torch.argmax(lm_head_out, dim=-1)[0][last_prefix_token + i]
        next_token_index = last_prefix_token + i + 1
        input_ids_pt[0][next_token_index] = next_token
        attention_mask[0][next_token_index] = 1


    generated_text_pt = tokenizer.decode(input_ids_pt[0][:next_token_index].numpy().tolist())
    logger.info(f"TT Generated text: {generated_text_tt}")
    logger.info(f"PT Generated text: {generated_text_pt}")

    assert generated_text_tt == generated_text_pt
