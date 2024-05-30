# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig, TTDeviceImage
from pybuda._C.backend_api import BackendType, BackendDevice, DeviceMode
from pybuda.verify.config import TestKind, NebulaGalaxy
from pybuda.pybudaglobal import TILE_DIM
from pybuda.utils import align_up_tile

import requests
import os
import torch.nn as nn
import pybuda
import torch
from PIL import Image

from transformers import (
    FuyuForCausalLM, 
    AutoTokenizer, 
    FuyuProcessor, 
    FuyuImageProcessor, 
    FuyuConfig,
    LogitsProcessorList
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import DynamicCache

def generate_fuyu_embedding(model, input_ids, image_patches, image_patches_indices):
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    patch_embeddings = model.vision_embed_tokens(image_patches.to(model.vision_embed_tokens.weight.dtype))
    inputs_embeds = model.gather_continuous_embeddings(
        word_embeddings=inputs_embeds,
        continuous_embeddings=patch_embeddings,
        image_patch_input_indices=image_patches_indices,
    )
    return inputs_embeds

class FuyuModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config 
 
    def forward(self, inputs_embeds):
        output_attentions = self.fuyu_config.output_attentions 
        use_cache = self.fuyu_config.use_cache
 
        # retrieve input_ids and inputs_embeds 
        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
 
        #device = input_ids.device if input_ids is not None else inputs_embeds.device
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0) 
        
        # PersimmonForCausalLM 
        output_hidden_states = self.fuyu_model.language_model.config.output_hidden_states 

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.fuyu_model.language_model.model(
            input_ids=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        return outputs
 

class FuyuModelImgDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config 

    def forward(self, inputs_embeds, attention_mask):
        batch_size, seq_length, hidden_dim = inputs_embeds.shape
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        hidden_states = inputs_embeds
        past_key_value = DynamicCache.from_legacy_cache()

        for idx, decoder_layer in enumerate(self.fuyu_model.language_model.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
            )

            hidden_states = layer_outputs[0]

        presents = past_key_value.to_legacy_cache()

        hidden_states = self.fuyu_model.language_model.model.final_layernorm(hidden_states)
        model_output = list()
        for key_state, value_state in presents:
            model_output.append(key_state)
            model_output.append(value_state)
        return hidden_states, *model_output

   
class FuyuModelTxtDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config 

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        batch_size, seq_length, _ = inputs_embeds.shape
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values[0].shape[-2]
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        hidden_states = inputs_embeds

        past_key_value_list = list()
        for idx in range(len(self.fuyu_model.language_model.model.layers)):
            pkv = tuple([past_key_values[(idx * 2) + j] for j in range(2)])
            past_key_value_list.append(pkv)

        past_key_value = DynamicCache.from_legacy_cache(tuple(past_key_value_list))

        presents = []
        for idx, decoder_layer in enumerate(self.fuyu_model.language_model.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
            )

            hidden_states = layer_outputs[0]

        presents = past_key_value.to_legacy_cache()

        hidden_states = self.fuyu_model.language_model.model.final_layernorm(hidden_states)
        model_output = list()
        for key_state, value_state in presents:
            model_output.append(key_state)
            model_output.append(value_state)

        return hidden_states, *model_output

def test_fuyu8b(test_device):
    pytest.skip("Already past-cache version is up")
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    #compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    #compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.convert_framework_params_to_tvm = False
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "FastCut"
    #compiler_cfg.amp_level = 2
    #compiler_cfg.default_dram_parameters = False
 
    config = FuyuConfig.from_pretrained("adept/fuyu-8b")
    config_dict = config.to_dict()
    config_dict['return_dict'] = False
    config_dict['use_cache'] = False
    config_dict['text_config']['num_hidden_layers'] = 1
    config = FuyuConfig(**config_dict)  

    # Load post-processing modules  (run on CPU)
    tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b") 
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Create PyBuda module from PyTorch model 
    fuyu_model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", config=config)
    #fuyu_model = FuyuForCausalLM(config=config) 
    model = FuyuModelWrapper(fuyu_model)
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_fuyu_8b", model)

    # Prepare inputs
    text_prompt = "Generate a coco-style caption.\n"
    image_path = "bus.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png
    image_pil = Image.open(image_path)
    model_inputs = processor(text=text_prompt, images=[image_pil], device="cpu", return_tensor='pt')
    inputs_embeds = generate_fuyu_embedding(fuyu_model, model_inputs['input_ids'], model_inputs['image_patches'][0], model_inputs['image_patches_indices'])
    inputs_embeds = inputs_embeds.clone().detach()
 
    # NOTE: it only runs language-model 
    verify_module(
        tt_model,  
        input_shapes=[(inputs_embeds.shape,),],
        inputs=[[inputs_embeds,],],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            verify_post_placer=False, 
        )
    )

def test_fuyu8b_past_cache(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Still under development")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.convert_framework_params_to_tvm = False
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.amp_level = 2
    compiler_cfg.default_dram_parameters = True 
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "FastCut"
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["TT_BACKEND_USE_PIPEGEN1"] = "1"
    os.environ["FUYU8B_FULL_LAYERS"] = "1" # flag to run the model wit full-layers, does not affect compile process

    if "FUYU8B_FULL_LAYERS" in os.environ and os.environ["FUYU8B_FULL_LAYERS"]:
        num_layers = 36
        for i in range(0, 80*num_layers, 80):
            compiler_cfg.balancer_op_override(f"matmul_{i+68}", "grid_shape", (1,8))
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{84*1024}"
    else:
        num_layers = 1
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"

    # Full-layer specific overrides
    for i in range(0, num_layers):
        if "FUYU8B_FULL_LAYERS" in os.environ and os.environ["FUYU8B_FULL_LAYERS"]:
            compiler_cfg.balancer_op_override(f"pt_fuyu8b_past_cache_img.output_concatenate_{i*80+41}_stack", "grid_shape", (1,1))
            compiler_cfg.balancer_op_override(f"pt_fuyu8b_past_cache_img.output_transpose_{i*80+53}_stack", "grid_shape", (1,1))
        else:
            compiler_cfg.balancer_op_override(f"pt_fuyu8b_past_cache_img_{num_layers}.output_concatenate_{i*80+41}_stack", "grid_shape", (1,1))
            compiler_cfg.balancer_op_override(f"pt_fuyu8b_past_cache_img_{num_layers}.output_transpose_{i*80+53}_stack", "grid_shape", (1,1))
        compiler_cfg.balancer_op_override(f"transpose_{i*80+91}.dc.sparse_matmul.4.lc2", "grid_shape", (8,1))
        compiler_cfg.balancer_op_override(f"transpose_{i*80+111}.dc.sparse_matmul.4.lc2", "grid_shape", (8,1))
        if num_layers > 1:
            compiler_cfg.balancer_op_override(f"transpose_{(i-1)*160+281}.dc.sparse_matmul.4.lc2", "grid_shape", (8,1))
    if "FUYU8B_FULL_LAYERS" in os.environ and os.environ["FUYU8B_FULL_LAYERS"]:
        for i in range(69):
            compiler_cfg.balancer_op_override(f"transpose_{i*80+262}.dc.sparse_matmul.4.lc2", "grid_shape", (2,1))
        for i in range(17):
            compiler_cfg.balancer_op_override(f"transpose_{i*160+3081}.dc.sparse_matmul.4.lc2", "grid_shape", (8,1))

    config = FuyuConfig.from_pretrained("adept/fuyu-8b")
    config_dict = config.to_dict()
    config_dict['return_dict'] = False
    config_dict['use_cache'] = False
    if "FUYU8B_FULL_LAYERS" in os.environ and os.environ["FUYU8B_FULL_LAYERS"]:
        pass 
    else:
        config_dict['text_config']['num_hidden_layers'] = num_layers
    config_dict['text_config']['max_position_embeddings'] = 448 #512
    config_dict['text_config']['pad_token_id'] = 0 # set '<unk>' equivalent id as pad-token-id of persimmon model (no default value is set)
    config = FuyuConfig(**config_dict)

    # Load post-processing modules  (run on CPU)
    tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b")
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Create PyBuda module from PyTorch model 
    fuyu_model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", config=config)

    # Prepare inputs
    text_prompt = "Generate a coco-style caption. "
    url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png" 
    image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    model_inputs = processor(text=text_prompt, images=[image_pil], device="cpu", return_tensor='pt')

    # Retrieve config numbers and logit function
    persimmon_config = fuyu_model.language_model.model.config 
    max_length = persimmon_config.max_position_embeddings 
    _, emb_seq_length = model_inputs['input_ids'].shape

    # Pad input_ids and image_patches_indices
    pad_inputs = True
    if pad_inputs:
        tmp_padding_token = 71128 # set \n as temporary padding string (does not matter)
        target_length = max_length - TILE_DIM
        org_length = model_inputs["input_ids"].shape[-1] 
        model_inputs["input_ids"] = torch.nn.functional.pad(model_inputs["input_ids"], (0, target_length-org_length), "constant", tmp_padding_token) 
        model_inputs["input_ids"][:, org_length-1] = tmp_padding_token
        model_inputs["input_ids"][:, -1] = 71122
        model_inputs['image_patches_indices'] = torch.nn.functional.pad(model_inputs['image_patches_indices'], (0, target_length+10-model_inputs['image_patches_indices'].shape[-1]), "constant", -1) 

    # Generate input embedding for the 1st iteration
    inputs_embeds = generate_fuyu_embedding(fuyu_model, model_inputs['input_ids'], model_inputs['image_patches'][0], model_inputs['image_patches_indices'])
    inputs_embeds = inputs_embeds.clone().detach() 
     
    # Obtain logit function
    logits_processor = fuyu_model._get_logits_processor(fuyu_model.generation_config, TILE_DIM, inputs_embeds, None, LogitsProcessorList())

    # Prepare compile-inputs for img-decoder 
    attention_mask = torch.zeros((1, max_length))
    attention_mask[0, :emb_seq_length] = 1
    img_attention_mask = torch.zeros((1, max_length-TILE_DIM), dtype=torch.bool)
    img_attention_mask[0, : emb_seq_length] = 1
    img_attention_mask = _prepare_4d_causal_attention_mask(img_attention_mask, (1, max_length-TILE_DIM), inputs_embeds, 0)
    img_decoder_inputs = [inputs_embeds, img_attention_mask]

    # Prepare compile-inputs for txt-decoder  
    input_ids = torch.zeros((1, TILE_DIM), dtype=torch.int) # 0 (corresponds to '<unk>')   
    inputs_embeds_dummy = torch.zeros((1, TILE_DIM, 4096))  # 4096 is hidden=state dim
    position_ids = torch.arange(TILE_DIM, dtype=torch.int).reshape(1, TILE_DIM) + align_up_tile(emb_seq_length) 
    first_current_index = max_length - TILE_DIM 
    past_cache_self_shape = (1, persimmon_config.num_attention_heads, max_length-TILE_DIM, persimmon_config.hidden_size // persimmon_config.num_attention_heads)
    txt_decoder_inputs = [inputs_embeds_dummy, attention_mask, position_ids.long()]
    for _ in range(len(fuyu_model.language_model.model.layers)):
        txt_decoder_inputs += [torch.zeros(past_cache_self_shape), torch.zeros(past_cache_self_shape),] 

    # Instantiate modules
    if "FUYU8B_FULL_LAYERS" in os.environ and os.environ["FUYU8B_FULL_LAYERS"]:
        img_decoder = pybuda.PyTorchModule("pt_fuyu8b_past_cache_img", FuyuModelImgDecoderWrapper(fuyu_model))  # feed inputs_embeds
        txt_decoder = pybuda.PyTorchModule("pt_fuyu8b_past_cache_txt", FuyuModelTxtDecoderWrapper(fuyu_model))  # feed inputs_embeds
    else:
        img_decoder = pybuda.PyTorchModule(f"pt_fuyu8b_past_cache_img_{num_layers}", FuyuModelImgDecoderWrapper(fuyu_model))  # feed inputs_embeds
        txt_decoder = pybuda.PyTorchModule(f"pt_fuyu8b_past_cache_txt_{num_layers}", FuyuModelTxtDecoderWrapper(fuyu_model))  # feed inputs_embeds

    # Place modules
    tt0 = pybuda.TTDevice(
        "tt0",
        devtype=test_device.devtype,
        device_mode=test_device.devmode,
        arch=test_device.arch,
        module=[img_decoder, txt_decoder])

    output_q = pybuda.initialize_pipeline(
        training=False,
        sample_inputs=((img_decoder_inputs), (txt_decoder_inputs))
    )

    ## TEST ## 
    generated_tokens = []
    current_token_index = align_up_tile(emb_seq_length)
    tokens_to_generate = 7 if test_device.devtype == BackendType.Silicon else 1
    for idx in range(tokens_to_generate):
        if idx == 0:
            tt0.set_active_subgraph(0)
            tt0.push_to_inputs([inputs_embeds, img_attention_mask])
            pybuda.run_generate(input_count=1, write_index=0)  # past-cache output to be MAX_LENGTH instead of 32
            ans = output_q.get()
            tt0.set_active_subgraph(1)
        else:
            tt0.push_to_inputs([inputs_embeds, attention_mask, position_ids])
            pybuda.run_generate(input_count=1, write_index=current_token_index // TILE_DIM,)
            ans = output_q.get() 

        hidden_states = ans[0].value().detach() 
        lm_head = fuyu_model.language_model.lm_head(hidden_states.float()).detach()
        _input_ids = torch.cat([torch.tensor([[1]]), input_ids[:, :current_token_index % TILE_DIM]], dim=-1)
        if idx == 0:
            tokens_scores = logits_processor(_input_ids, lm_head[:, current_token_index-1, :])
        else:
            tokens_scores = logits_processor(_input_ids, lm_head[:, (current_token_index-1) % TILE_DIM, :])
        next_token = torch.argmax(tokens_scores, dim=-1).item()
        generated_tokens.append(next_token)

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            attention_mask[0, :current_token_index] = 1
            attention_mask[0, first_current_index:] = 0
            position_ids = position_ids + TILE_DIM
            input_ids[0, :] = 0

        input_ids[0, (current_token_index-1) % TILE_DIM] = next_token
        attention_mask[0, first_current_index + ((current_token_index-1) % TILE_DIM)] = 1
        inputs_embeds = fuyu_model.language_model.model.embed_tokens(input_ids).detach()

    # Post-process
    print("generated-tokens = ", generated_tokens)
    generated_text = processor.batch_decode(torch.tensor([generated_tokens]), skip_special_tokens=True)
    print("generated-text = ", generated_text)

