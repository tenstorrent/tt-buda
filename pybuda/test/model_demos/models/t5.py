# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from test.utils import download_model
from pybuda.pybudaglobal import TILE_DIM
#import queue
import os
import pybuda
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


class T5_encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5 = model

    def forward(self, input_ids, attention_mask):
        return self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
    
class T5_decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5 = model

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, decoder_encoder_attention_mask, *past_key_values):
        presents = []
        pkv = []
        for i, _ in enumerate(self.t5.decoder.block):
            pkv.append(tuple([past_key_values[(i * 4) + j] for j in range(4)]))

        outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=decoder_encoder_attention_mask,
            past_key_values = pkv,
        )
        sequence_output = outputs[0]
        presents = outputs[1]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)

        lm_logits = self.t5.lm_head(sequence_output)    
        return lm_logits, *presents



def generate_t5_past_cache_enc_dec(test_device, variant): 
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "64"
    os.environ["PYBUDA_ROTATE_PAST_CACHE_PARAMS"] = "1"
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.enable_amp_light()
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_link_past_cache_ios = True
 
    model_name = variant
    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False 
    max_length = config_dict["n_positions"]

    config = T5Config(**config_dict)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    num_blocks = len(model.decoder.block)
    if "n_layers" in locals():
        num_blocks = n_layers
    for i in range(num_blocks):
        pybuda.config.override_op_size(f"t5.decoder.block.{i}.layer.0.SelfAttention.k.weight_cache_nop", [1, 1])
        pybuda.config.override_op_size(f"t5.decoder.block.{i}.layer.0.SelfAttention.v.weight_cache_nop", [1, 1])
        pybuda.config.override_t_stream_shape(f"t5.decoder.block.{i}.layer.0.SelfAttention.k.weight_cache_nop", [15, 1])
        pybuda.config.override_t_stream_shape(f"t5.decoder.block.{i}.layer.0.SelfAttention.v.weight_cache_nop", [15, 1])

    input_length = 64
    input_text = "translate English to German: The house is wonderful. We have really enjoyed living here for the past eight years. The only problem that I have with it is that it is too small and the parks are not very close."
    encoder_inptus = tokenizer(input_text, return_tensors="pt", max_length=input_length, pad_to_max_length=True, truncation=True)
    input_ids = encoder_inptus["input_ids"].int()
    encoder_attention_mask = encoder_inptus["attention_mask"].float()

    encoder_last_hidden_state_shape = (1, input_length, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)
    decoder_input_ids = torch.zeros((1, TILE_DIM), dtype=torch.int)
    decoder_attention_mask = torch.zeros((1, max_length))
    
    enc_past_cache_self_shape = (1, config.num_heads, max_length-32, config.d_kv)
    enc_past_cache_cross_shape = (1, 1, 1, 1)
    decoder_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask]
    for _ in range(num_blocks):
        decoder_ca_inputs +=  [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    enc_past_cache_cross_shape = (1, config.num_heads, input_length, config.d_kv)
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask]
    for _ in range(num_blocks):
        decoder_no_ca_inputs += [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    encoder_module = pybuda.PyTorchModule("T5_encoder", T5_encoder(model))
    decoder_module_cross_attention = pybuda.PyTorchModule("T5_decoder_with_ca", T5_decoder(model))
    decoder_module_no_cross_attention = pybuda.PyTorchModule("T5_decoder_no_ca", T5_decoder(model)) 

    first_current_index = max_length - TILE_DIM
    decoder_attention_mask[0, first_current_index] = 1
    
    modules = {}
    modules["tt"] = [encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention]
    compile_inputs = (
        (input_ids, encoder_attention_mask),
        (decoder_ca_inputs),
        (decoder_no_ca_inputs),
    ) 
    
    return modules, [input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask], {"compile_inputs": compile_inputs, "max_length": input_length, "first_current_index": first_current_index, "pad_token_id": tokenizer.pad_token_id}
 

