# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from test.backend.models.test_bert import get_relaxed_atol_pcc

from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    VerifyConfig,
    run_generate,
)
import pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

from typing import Optional, Tuple, Union
import math
from loguru import logger

import os

input_shapes = [(1, 64, 768)]


def test_pt_gpt2_tokengen():
    torch.set_printoptions(linewidth=200)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    pad_token = tokenizer(tokenizer.pad_token)["input_ids"][0]

    past_cache_length = 30
    run_length = 6

    total_length = past_cache_length + run_length
    inputs = tokenizer("The", max_length=past_cache_length, pad_to_max_length=True, truncation=True, return_tensors="pt")
    input_ids = torch.tensor(inputs["input_ids"])
    input_ids[:][:] = pad_token
    past_key_values = model(input_ids, return_dict=False)[1]

    next_key_values = past_key_values
    past_key_values = list(next_key_values)
    for block_idx, block in enumerate(next_key_values):
        past_key_values[block_idx] = list(block)
        for kv_idx, kv in enumerate(block):
            past_key_values[block_idx][kv_idx] = torch.zeros_like(kv)

    prefix_text = "My name is Bert,"
    inputs = tokenizer(prefix_text, max_length=run_length, pad_to_max_length=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    last_prefix_token =  int((attention_mask[0] == 0).nonzero()[0][0]) - 1
    tokens_to_generate = 30

    full_attention_mask = torch.zeros(total_length).int().unsqueeze(0)
    full_attention_mask[:,past_cache_length:] = torch.tensor(inputs["attention_mask"]).int()

    all_generated_tokens = []

    past_length = 0
    print("")
    for i in range(last_prefix_token, last_prefix_token + tokens_to_generate):
        position_ids = torch.arange(past_length, past_length + run_length)
        lm_head_out, next_key_values = model(input_ids, attention_mask=full_attention_mask, past_key_values=past_key_values, position_ids=position_ids, return_dict=False)

        next_token = torch.argmax(lm_head_out, dim=-1)[0][i % run_length]
        all_generated_tokens.append(next_token)

        next_token_index = (i + 1) % run_length
        if next_token_index == 0:
            tile_index = ((i + 1) // run_length) - 1
            past_length += run_length
            input_ids[:][:] = pad_token
            pce = i + 1
            pcb = 0
            full_attention_mask[:, pcb:pce] = 1
            full_attention_mask[:, -run_length:] = 0
            print(full_attention_mask)

            past_key_values = list(next_key_values)
            for block_idx, block in enumerate(next_key_values):
                past_key_values[block_idx] = list(block)
                for kv_idx, kv in enumerate(block):
                    past_key_values[block_idx][kv_idx][:, :, tile_index*run_length:(tile_index+1)*run_length, :] = kv[:, :, -run_length:, :]
                    past_key_values[block_idx][kv_idx] = past_key_values[block_idx][kv_idx].narrow(2, 0, past_cache_length)
            print(abs(past_key_values[0][0][:, 0, :, 0].int()))
        input_ids[0][next_token_index] = next_token
        full_attention_mask[0][past_cache_length + next_token_index] = 1

    generated_text_pt = tokenizer.decode(all_generated_tokens)
    logger.info(f"PT Generated text: {generated_text_pt}")

def test_tvm_gpt2_attention_with_past_cache(test_device):
    hidden_size = 768
    num_heads = 12
    max_seq_len = 512
    
    class AttentionWrapper(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn
        
        def forward(self, hidden_states, attention_mask, key_past, value_past):
            key_past = self.attn._split_heads(key_past, 12, 64)
            value_past = self.attn._split_heads(value_past, 12, 64)
            layer_past = (key_past, value_past)
            hidden_states, (key_past, value_past) = self.attn(hidden_states, layer_past=layer_past, use_cache=True, attention_mask=attention_mask)
            key_present = key_past[:, :, -32:, :]
            key_present = self.attn._merge_heads(key_present, 12, 64)
            value_present = value_past[:, :, -32:, :]
            value_present = self.attn._merge_heads(value_present, 12, 64)
            return hidden_states, key_present, value_present
    
    torch.manual_seed(52)
    torch_mod = AttentionWrapper(GPT2Model.from_pretrained("gpt2").h[0].attn)
    layer_past_shape = (1, 480, 768)
    mod = PyTorchModule("gpt2_cached_attn", torch_mod)

    hidden_states_shape = (1, 32, hidden_size)
    attention_mask_shape = (1, 1, 1, max_seq_len)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.loopback_outputs = {"tensor_1": 1, "tensor_5": 2}

    tt0 = TTDevice("tt0", devtype=test_device.devtype)
    tt0.place_module(mod)
    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=(torch.rand(hidden_states_shape), torch.rand(attention_mask_shape), torch.zeros(layer_past_shape), torch.zeros(layer_past_shape)), _verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            verify_all=True,
        ))
    print(pybuda.get_parameter_checkpoint()[0]['tensor_1'])
    tt0.push_to_inputs((torch.rand(hidden_states_shape), torch.rand(attention_mask_shape), ))
    pybuda.run_generate(input_count=1, write_index=0)
    pk = pybuda.get_parameter_checkpoint()[0]['tensor_1'].value()
    ans = output_q.get(timeout = 0.5)
    print(pybuda.get_parameter_checkpoint()[0]['tensor_1'])
    tt0.push_to_inputs((torch.rand(hidden_states_shape), torch.rand(attention_mask_shape), ))
    pybuda.run_generate(input_count=1, write_index=1)
    print(pybuda.get_parameter_checkpoint()[0]['tensor_1'])
    tt0.push_to_inputs((torch.rand(hidden_states_shape), torch.rand(attention_mask_shape), ))
    pybuda.run_generate(input_count=1, write_index=2)
    print(pybuda.get_parameter_checkpoint()[0]['tensor_1'])

class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gpt2 = model.transformer

    def forward(self, input_ids, attention_mask, position_ids, *kv):
        inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        return hidden_states, extended_attention_mask, *kv

class BlocksWrapper(torch.nn.Module):
    def __init__(self, model, num_blocks):
        super().__init__()
        self.gpt2 = model.transformer
        self.num_blocks = num_blocks

    def forward(self, hidden_states, extended_attention_mask, *kv):
        presents = []
        for i, block in enumerate(self.gpt2.h[:self.num_blocks]):
            past_key = kv[i * 2]
            past_value = kv[(i * 2) + 1]
            past_key = self.gpt2.h[0].attn._split_heads(past_key, 12, 64)
            past_value = self.gpt2.h[0].attn._split_heads(past_value, 12, 64)
            layer_past = (past_key, past_value)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=extended_attention_mask,
                use_cache=True,
            )
            hidden_states = outputs[0]
            key_present = outputs[1][0][:, :, -32:, :]
            key_present = self.gpt2.h[0].attn._merge_heads(key_present, 12, 64)
            presents.append(key_present)
            value_present = outputs[1][1][:, :, -32:, :]
            value_present = self.gpt2.h[0].attn._merge_heads(value_present, 12, 64)
            presents.append(value_present)
        hidden_states = self.gpt2.ln_f(hidden_states)
        return hidden_states, *presents

class LMHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden_states, *kv):
        return self.lm_head(hidden_states)

def test_gpt2_past_cache(test_device):
    if not test_device.is_silicon():
        pytest.skip() # too long for post-commit

    compiler_cfg = _get_global_compiler_config()
    num_blocks = 12
    if num_blocks == 12:
        compiler_cfg.loopback_outputs = {"tensor_1": 1, "tensor_5": 2,
                                        "tensor_33": 3, "tensor_37": 4,
                                        "tensor_65": 5, "tensor_69": 6,
                                        "tensor_97": 7, "tensor_101": 8,
                                        "tensor_129": 9, "tensor_133": 10,
                                        "tensor_161": 11, "tensor_165": 12,
                                        "tensor_193": 13, "tensor_197": 14,
                                        "tensor_225": 15, "tensor_229": 16,
                                        "tensor_257": 17, "tensor_261": 18,
                                        "tensor_289": 19, "tensor_293": 20,
                                        "tensor_321": 21, "tensor_325": 22,
                                        "tensor_353": 23, "tensor_357": 24}
    else:
        compiler_cfg.loopback_outputs = {"past_key_1": 1, "past_value_1": 2, "past_key": 3, "past_value": 4}

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    embeddings = EmbWrapper(model)
    blocks = BlocksWrapper(model, num_blocks=num_blocks)
    lm_head = LMHeadWrapper(model)


    past_length = 0
    run_length = 32
    pad_token = tokenizer(tokenizer.pad_token)["input_ids"][0]
    prefix_text = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains."
    inputs = tokenizer(prefix_text, max_length=run_length, pad_to_max_length=True, truncation=True)
    prefix_text = ""
    input_ids_tt = torch.tensor(inputs["input_ids"]).int().unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"]).int().unsqueeze(0)
    attention_mask = torch.cat((torch.zeros(1, 480), attention_mask), -1)
    position_ids = torch.arange(past_length, past_length + run_length)

    last_prefix_token =  inputs["attention_mask"].index(0) - 1 
    tokens_to_generate = 480

    cpu0 = pybuda.CPUDevice("cpu0", module=PyTorchModule("gpt2_embeddings", embeddings))
    tt1 = pybuda.TTDevice("tt1", 
            devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("gpt2_blocks", blocks))
    cpu1 = pybuda.CPUDevice("cpu1", module=PyTorchModule("gpt2_lm_head", lm_head))

    layer_past_shape = (1, 480, 768)
    inputs = (input_ids_tt, attention_mask, position_ids)
    for _ in range(num_blocks):
        inputs += (torch.zeros(layer_past_shape), torch.zeros(layer_past_shape))

    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=inputs,)
    write_index = 0
    current_token_index = last_prefix_token
    for i in range(tokens_to_generate):
        position_ids = torch.arange(past_length, past_length + run_length)
        cpu0.push_to_inputs((input_ids_tt, attention_mask, position_ids))
        pybuda.run_generate(input_count=1, write_index=write_index)
        outputs = output_q.get()
        lm_head_out = outputs[0].value().detach()
        k = 10
        top_k_probs, top_k_ids = torch.topk(lm_head_out[0,current_token_index], k=k)
        next_token = top_k_ids[torch.randint(k-1, (1, ))]
        # next_token = torch.argmax(lm_head_out, dim=-1)[0][current_token_index]
        current_token_index += 1
        if current_token_index == 32:
            past_length += run_length
            current_token_index = 0
            attention_mask[0][write_index * 32 : (write_index + 1) * 32] = 1
            attention_mask[0][-32:] = 0
            write_index += 1
            prefix_text += tokenizer.decode(input_ids_tt[0][:].numpy().tolist())
            input_ids_tt[:][:] = pad_token
        input_ids_tt[0][current_token_index] = next_token
        scrubbed_input = input_ids_tt[input_ids_tt != pad_token]
        print(f"Generated text: {tokenizer.decode(scrubbed_input.numpy().tolist())}")
        attention_mask[0][480 + current_token_index] = 1

    print(f"Generated text: {prefix_text}")
    
    # prefix_text = "My name is Ljubisa, and I am"
    # inputs = tokenizer(prefix_text, max_length=64, pad_to_max_length=True, truncation=True)
    # input_ids_pt = torch.tensor(inputs["input_ids"]).int().unsqueeze(0)
    # attention_mask = torch.tensor(inputs["attention_mask"]).int().unsqueeze(0)

    # for i in range(tokens_to_generate):
    #     embedding_output = embeddings(input_ids_pt, attention_mask)
    #     model_output = blocks(*embedding_output)
    #     lm_head_out = lm_head(model_output)
    #     next_token = torch.argmax(lm_head_out, dim=-1)[0][last_prefix_token + i]
    #     next_token_index = last_prefix_token + i + 1
    #     input_ids_pt[0][next_token_index] = next_token
    #     attention_mask[0][next_token_index] = 1

def test_tvm_past_cache_generate(test_device):
    class PastCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, y, key_past):
            key_past = key_past + y
            key_past_sliced = key_past[:, :, -32:, :]
            return key_past_sliced, key_past_sliced + 1
    
    torch_mod = PastCache()
    mod = PyTorchModule("cached_attn", torch_mod)
    tt0 = TTDevice("tt0", devtype=test_device.devtype)
    tt0.place_module(mod)


    single_cache_line = (1, 1, 32, 32)
    layer_past_shape =  (1, 1, 192, 32)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.loopback_outputs = {"key_past_1": 0}

    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=(torch.rand(1), torch.zeros(layer_past_shape)), _verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
        ))
    print(pybuda.get_parameter_checkpoint())
    tt0.push_to_inputs((torch.rand(1), ))
    tt0.push_to_inputs((torch.rand(1), ))
    pybuda.run_generate(input_count=2, tokens_per_iter=32, token_id=0)
    print(pybuda.get_parameter_checkpoint())
    ans = output_q.get()
    tt0.push_to_inputs((torch.rand(1), ))
    pybuda.run_generate(input_count=1, tokens_per_iter=31, token_id=64)
    ans = output_q.get()
    print(pybuda.get_parameter_checkpoint())
    tt0.push_to_inputs((torch.rand(1), ))
    tt0.push_to_inputs((torch.rand(1), ))
    pybuda.run_generate(input_count=2, tokens_per_iter=1, token_id=95)
    ans = output_q.get()
    print(pybuda.get_parameter_checkpoint())


def test_past_cache_prefill_generate(test_device):
    class PastCache_attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm = torch.nn.Parameter(torch.rand(32, 32))
        
        def forward(self, input_prefill):
            out = torch.matmul(input_prefill, self.mm)
            return out 

    
    class PastCache_prefill(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, input_gen, prefill_output):

            key_past = input_gen + prefill_output
            key_past_sliced = key_past[:, :, -32:, :]
            return key_past_sliced
    torch_mod_0 = PastCache_attn()
    torch_mod_1 = PastCache_prefill()

    mod_0 = PyTorchModule("cached_attn", torch_mod_0)
    mod_1 = PyTorchModule("cached_prefill", torch_mod_1)

    tt0 = TTDevice("tt0", devtype=test_device.devtype)
    tt0.place_module(mod_0)
    tt0.place_module(mod_1)


    input_prefil_shape = (1, 1, 480, 32)
    input_generate_shape = (1,)
    past_shape = (1, 1, 480, 32)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.balancer_op_override("matmul_3_output_nop_0", "t_stream_shape", (15,1))
    # compiler_cfg.loopback_outputs = {"prefill_output": (0, 1)}

    output_q = pybuda.initialize_pipeline(
            training=False,
            sample_inputs=((torch.rand(input_prefil_shape),), (torch.rand(input_generate_shape), torch.rand(past_shape),),), 
            _verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
            ).disabled()
        )
    tt0.set_active_subgraph(0)
    tt0.push_to_inputs((torch.rand(input_prefil_shape), ))
    pybuda.run_forward()

    tt0.set_active_subgraph(1)
    tt0.push_to_inputs((torch.rand(input_generate_shape), torch.rand(past_shape),))
    pybuda.run_forward()


@pytest.mark.skip(reason="Tested with fallback")
def test_tvm_gpt2_block(test_kind, test_device):
    # Training without TVM constant prop will result in the following error in placer
    #   RuntimeError: trying to place bw_in0_gpt2_block.attn_c_attn_weight_combine_add_0_transpose_nop 
    #   and exceeded max-placement-attempts: grid_shape: (2, 8, original context.start=(.row=5, .col = 5)
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    model = GPT2Model.from_pretrained("gpt2")
    mod = PyTorchModule("gpt2_block", model.h[0])
    input_shape = (1, 64, 768)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"}
    
    relative_atol = 0.4 if test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.9 if test_device.devtype == BackendType.Silicon else 0.99

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"c_attn.bias_1"},
            relative_atol=relative_atol,
            pcc=pcc,
        )
    )

@pytest.mark.skip(reason="Tested with fallback")
def test_tvm_gpt2_blocks(test_device):
    class ListWrapper(torch.nn.Module):
        def __init__(self, module_list):
            super().__init__()
            self.module_list = module_list

        def forward(self, hidden_states):
            for module in self.module_list:
                hidden_states = module(hidden_states)[0]

            return hidden_states

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"}

    input_shape = (1, 64, 768)
    model = GPT2Model.from_pretrained("gpt2")

    torch_mod = ListWrapper(model.h)
    mod = PyTorchModule("gpt2", torch_mod)

    torch.manual_seed(42)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
        uniform_inputs=True,
    )

def test_new_gelu(test_device):
    class NewGELUActivation(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
        the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, input):
            return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    torch_mod = NewGELUActivation()
    mod = PyTorchModule("new_gelu", torch_mod)

    input_shape = (1, 64, 3072)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        )
    )

def test_gelu(test_device):
    class GELUActivation(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
        the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, input):
            return torch.nn.functional.gelu(input)

    torch_mod = GELUActivation()
    mod = PyTorchModule("gelu", torch_mod)

    input_shape = (1, 64, 3072)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        )
    )


from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
def test_tvm_gpt2_fallback(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"} 

    input_shape = (1, 768)
   
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 2
    # config.use_cache = False
    config.return_dict = False

    model = GPT2LMHeadModel(config)

    mod = PyTorchModule("gpt2", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"c_attn.bias"},
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )

from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
def test_tvm_gpt2_lmhead(test_kind, test_device):
    class LMHeadWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states):
            hidden_states = model.transformer.h[0](hidden_states)[0]
            lm_logits = self.model.lm_head(hidden_states)

            return lm_logits

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"}


    input_shape = input_shape = (1, 64, 768)
   
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 1
    config.use_cache = False
    config.return_dict = False


    model = GPT2LMHeadModel(config)

    mod = PyTorchModule("gpt2", LMHeadWrapper(model))
    
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"c_attn.bias"},
            relative_atol=relative_atol,
            pcc=pcc,
        ),
        uniform_inputs=True,
    )

class SpliceUnit(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, cache, line):
        # cache: 1 x 1 x 32 x 32
        # line:  1 x 1 x 32 x 1
        x = cache[...,1:]
        out = torch.cat((x, line), dim=-1)
        return out

def test_splice(test_device):
    import pybuda
    mod = SpliceUnit()
    pb_mod = pybuda.PyTorchModule('splice', mod)

    verify_module(pb_mod, [(1, 1, 32, 32), (1, 1, 32, 1)],
            VerifyConfig(test_kind=TestKind.INFERENCE,
                arch=test_device.arch,
                devtype=test_device.devtype,
            ),
    )
