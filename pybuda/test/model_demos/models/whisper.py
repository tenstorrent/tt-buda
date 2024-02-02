# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch
from transformers import (
    AutoProcessor,
    WhisperConfig,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import pybuda
from test.utils import download_model
from pybuda.pybudaglobal import TILE_DIM
from pybuda.config import _get_global_compiler_config
from pybuda._C.backend_api import BackendType, BackendDevice


class Whisper_encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        return self.model.model.encoder(input_features=input_features)


class Whisper_decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds, *past_key_values):
        presents = []
        pkv = []

        input_embeds = self.model.model.decoder.embed_tokens(decoder_input_ids)
        hidden_states = input_embeds + position_embeds

        attention_mask = _prepare_4d_causal_attention_mask(decoder_attention_mask, decoder_input_ids.size(), input_embeds, past_key_values[0].shape[2])

        presents = []
        for i, decoder_layer in enumerate(self.model.model.decoder.layers):
            pkv = tuple([past_key_values[(i * 4) + j] for j in range(4)])


            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_last_hidden_state,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=pkv,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]
            presents.append(layer_outputs[1])

        hidden_states = self.model.model.decoder.layer_norm(hidden_states)
        lm_logits = self.model.proj_out(hidden_states)

        return lm_logits, *presents


def generate_model_whisper_decoder_past_cache(test_device, variant):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"


    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.amp_level = 1
        os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for LM head to output queue (perf)
        os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
        os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
        os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
        os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"

        os.environ["PYBUDA_NOP_ON_DIRECT_SHORT_PATH"] = "1"
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "23000"
        os.environ["PYBUDA_SKIP_SMALL_UKT"] = "1"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
        if variant in ["openai/whisper-base", "openai/whisper-medium", "openai/whisper-large"]:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    
    # pybuda.set_configuration_options(performance_trace=pybuda.PerfTraceLevel.VERBOSE)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    config = WhisperConfig.from_pretrained(variant)
    max_length = config.max_length
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        return_dict=False,
    )
    feature_extractor = download_model(WhisperFeatureExtractor.from_pretrained, variant)
    tokenizer = WhisperTokenizer.from_pretrained(variant)
    decoder_module_no_cross_attention = pybuda.PyTorchModule("Whisper_decoder_no_ca", Whisper_decoder(model))

    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    sequence_length = config.max_source_positions
    sequence_length = 1536
    encoder_last_hidden_state_shape = (1, sequence_length, config.d_model)
    encoder_last_hidden_state = torch.rand(encoder_last_hidden_state_shape)

    decoder_attention_mask = torch.rand((1, max_length))
    decoder_input_ids = torch.ones((1, TILE_DIM), dtype=torch.int) * tokenizer.pad_token_id
    position_embeds = torch.rand((TILE_DIM, config.d_model))
    enc_past_cache_self_shape = (1, config.decoder_attention_heads, max_length-TILE_DIM, config.d_model // config.decoder_attention_heads)
    enc_past_cache_cross_shape = (1, config.decoder_attention_heads, sequence_length, config.d_model // config.decoder_attention_heads)
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_no_ca_inputs += [torch.rand(enc_past_cache_self_shape), torch.rand(enc_past_cache_self_shape),
                   torch.rand(enc_past_cache_cross_shape), torch.rand(enc_past_cache_cross_shape)]

    return decoder_module_no_cross_attention, [decoder_input_ids, decoder_attention_mask, position_embeds], {"compile_inputs": decoder_no_ca_inputs, "max_length": max_length}


# check the name later # enc-dec
def generate_model_whisper_enc_dec(test_device, variant):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.amp_level = 1
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.backend_opt_level = 3
    #compiler_cfg.enable_auto_fusing = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for LM head to output queue (perf)
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER_THRESHOLD_TILES"] = "1536"

    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
    os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    os.environ["TT_BACKEND_PROFILER"] = "1"
    os.environ["PYBUDA_NOP_ON_DIRECT_SHORT_PATH"] = "1"

    if variant == "openai/whisper-base":
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "None"
        compiler_cfg.enable_auto_fusing = False

    if variant == "openai/whisper-medium" or variant == "openai/whisper-large":
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "0"

    run_encoder_on_tt = ("tiny" in variant) or ("base" in variant) or ("small" in variant)

    pad_model = True
    # pybuda.set_configuration_options(performance_trace=pybuda.PerfTraceLevel.VERBOSE)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    config = WhisperConfig.from_pretrained(variant)
    config.return_dict = False
    if pad_model:
        config.max_source_positions = 1536
    max_length = config.max_length
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        ignore_mismatched_sizes=True,
        config=config,
    )
    if pad_model:
        unpadded_model = WhisperForConditionalGeneration.from_pretrained(variant)
        padded_param = torch.nn.functional.pad(unpadded_model.model.encoder.embed_positions.weight.data, (0,0,0,36))
        model.model.encoder.embed_positions.weight.data = padded_param

    feature_extractor = download_model(WhisperFeatureExtractor.from_pretrained, variant)
    tokenizer = WhisperTokenizer.from_pretrained(variant)
    encoder_module = pybuda.PyTorchModule("Whisper_encoder", Whisper_encoder(model))
    decoder_module_cross_attention = pybuda.PyTorchModule("Whisper_decoder_with_ca", Whisper_decoder(model))
    decoder_module_no_cross_attention = pybuda.PyTorchModule("Whisper_decoder_no_ca", Whisper_decoder(model))

    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]
    inputs = feature_extractor(sample_audio, return_tensors="pt") 
 
    if pad_model:
        input_features = torch.nn.functional.pad(inputs.input_features, (0, 72, 0, 0))
    else:
        input_features = inputs.input_features

    encoder_last_hidden_state_shape = (1, config.max_source_positions, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)

    #logits_processor = model._get_logits_processor(model.generation_config, TILE_DIM, input_features, None, LogitsProcessorList())
    decoder_attention_mask = torch.zeros((1, max_length))
    decoder_input_ids = torch.ones((1, TILE_DIM), dtype=torch.int) * tokenizer.pad_token_id
    first_current_index = max_length - TILE_DIM
    position_embeds = torch.zeros((TILE_DIM, config.d_model))
    enc_past_cache_self_shape = (1, config.decoder_attention_heads, max_length-TILE_DIM, config.d_model // config.decoder_attention_heads)
    enc_past_cache_cross_shape = (1, 1, 1, 1)

    decoder_with_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_with_ca_inputs += [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    dec = Whisper_decoder(model)
    dec(*decoder_with_ca_inputs)
    enc_past_cache_cross_shape = (1, config.decoder_attention_heads, config.max_source_positions, config.d_model // config.decoder_attention_heads)
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_no_ca_inputs += [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]
 
    inputs = feature_extractor(sample_audio, return_tensors="pt")

    if pad_model:
        input_features = torch.nn.functional.pad(inputs.input_features, (0, 72, 0, 0))
    else:
        input_features = inputs.input_features
    decoder_attention_mask = torch.zeros((1, max_length))
    decoder_input_ids[0, 0] = tokenizer.encode('<|startoftranscript|>')[0]
    decoder_attention_mask[0, first_current_index] = 1
    current_token_index = 0

    prefix_tokens = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    for idx, token in prefix_tokens:
        decoder_input_ids[0, idx] = token
        decoder_attention_mask[0, first_current_index + idx] = 1
        current_token_index = idx

    position_ids = torch.arange(32, dtype=torch.long)
    position_embeds = model.model.decoder.embed_positions.weight[position_ids]
 
    modules = [encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention]
    compile_inputs = ((input_features,), (decoder_with_ca_inputs), (decoder_no_ca_inputs),)

    return modules, [input_features, position_embeds, decoder_input_ids, decoder_attention_mask], {"compile_inputs": compile_inputs, "max_length": 64, "write_index": (current_token_index//TILE_DIM)}
