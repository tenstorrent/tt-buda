# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    LogitsProcessorList
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import pybuda
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType
from pybuda import PyTorchModule, VerifyConfig
from pybuda.config import _get_global_compiler_config
from test.utils import download_model

from pybuda.pybudaglobal import TILE_DIM

variants = [
    "openai/whisper-tiny",
    # "openai/whisper-base",
    # "openai/whisper-small",
    # "openai/whisper-medium",
    # "openai/whisper-large",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper_encoder(test_device, variant):
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.amp_level = 1
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    pcc = 0.93 if test_device.devtype == BackendType.Silicon else 0.99

    if variant == "openai/whisper-small" or variant == "openai/whisper-medium" or variant == "openai/whisper-large":
        os.environ["PYBUDA_PAD_MM"] = "{47:48}"

    if variant == "openai/whisper-tiny":
        os.environ["PYBUDA_TEMP_FIX_2351"] = "1"

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features):
            enc_out = self.model.model.encoder(
                input_features
            )

            return enc_out[0]

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    framework_model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        use_cache=False,
        return_dict=False,
    )
    
    framework_model = Wrapper(framework_model)
    pybuda_model = PyTorchModule("pt_whisper", framework_model)

    # Load and preprocess sample audio
    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Sanity run
    out = framework_model(input_features)

    verify_module(
        pybuda_model,
        [
            (input_features.shape),
        ],
        inputs=[
            (input_features),
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper_decoder(test_device, variant):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.amp_level = 1
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for LM head to output queue (perf)
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

            self.decoder_attention_mask = torch.ones((1, 1))

        def forward(self, decoder_input_ids, encoder_hidden_states):
            dec_out = self.model.model.decoder(
                decoder_input_ids,
                self.decoder_attention_mask,
                encoder_hidden_states,
            )
            lin_out = self.model.proj_out(dec_out[0])

            return lin_out

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    model_config = WhisperConfig()

    # Reduce size of model for testing
    # model_config.use_cache = False
    # model_config.return_dict = False
    # model_config.decoder_attention_heads = 1
    # model_config.decoder_layers = 1
    # model_config.encoder_attention_heads = 1
    # model_config.encoder_layers = 1
    # model_config.num_hidden_layers = 1
    # model_config.d_model = 384
    # framework_model = download_model(
    #     WhisperForConditionalGeneration.from_pretrained,
    #     variant,
    #     config=model_config,
    # )

    framework_model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        use_cache=False,
        return_dict=False,
    )

    framework_model = Wrapper(framework_model)
    pybuda_model = PyTorchModule("pt_whisper", framework_model)

    # Load and preprocess sample audio
    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_input_ids = torch.tensor([[1, 1]]) * model_config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(torch.int32)
    encoder_outputs = framework_model.model.model.encoder(input_features)[0].detach()
    encoder_outputs = encoder_outputs.to(torch.float32)

    # Sanity run
    out = framework_model(decoder_input_ids, encoder_outputs)

    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        pybuda_model,
        [
            (decoder_input_ids.shape, encoder_outputs.shape),
        ],
        inputs=[
            (decoder_input_ids, encoder_outputs),
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )


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

@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper_enc_dec(test_device, variant):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.amp_level = 1
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for LM head to output queue (perf)
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
    os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    os.environ["TT_BACKEND_PROFILER"] = "1"

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
    encoder_module = pybuda.PyTorchModule("Whisper_encoder", Whisper_encoder(model))
    decoder_module_cross_attention = pybuda.PyTorchModule("Whisper_decoder_with_ca", Whisper_decoder(model))
    decoder_module_no_cross_attention = pybuda.PyTorchModule("Whisper_decoder_no_ca", Whisper_decoder(model))

    for i in range(config.decoder_layers):
        pybuda.config.override_t_stream_shape(f"model.model.decoder.layers.{i}.self_attn.k_proj.weight_cache_nop", [13, 1])
        pybuda.config.override_t_stream_shape(f"model.model.decoder.layers.{i}.self_attn.v_proj.weight_cache_nop", [13, 1])

    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")

    input_features = inputs.input_features

    encoder_last_hidden_state_shape = (1, config.max_source_positions, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)

    logits_processor = model._get_logits_processor(model.generation_config, TILE_DIM, input_features, None, LogitsProcessorList())
    sequence_length = 1500
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
    enc_past_cache_cross_shape = (1, config.decoder_attention_heads, sequence_length, config.d_model // config.decoder_attention_heads)
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_no_ca_inputs += [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    tt0 = pybuda.TTDevice(
        "tt0", 
        devtype=test_device.devtype, 
        arch=test_device.arch, 
        module=[decoder_module_cross_attention, decoder_module_no_cross_attention])
        # module=[encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention])

    output_q = pybuda.initialize_pipeline(
        training=False,
        sample_inputs=(
            # (input_features,),
            (decoder_with_ca_inputs),
            (decoder_no_ca_inputs),
        ))

    import time
    decoder_attention_mask = torch.zeros((1, max_length))
    decoder_input_ids[0, 0] = tokenizer.encode('<|startoftranscript|>')[0]
    decoder_attention_mask[0, first_current_index] = 1
    current_token_index = 0

    prefix_tokens = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    for idx, token in prefix_tokens:
        decoder_input_ids[0, idx] = token
        decoder_attention_mask[0, first_current_index + idx] = 1
        current_token_index = idx

    # encoder hangs, for now run on cpu
    encoder_last_hidden_state = model.model.encoder(input_features)[0].detach()
    start = time.time()
    # tt0.set_active_subgraph(0)
    # tt0.push_to_inputs((input_features, ))
    # pybuda.run_forward()
    # ans = output_q.get()
    # encoder_last_hidden_state = ans[0].value().detach()
    generated_tokens = []
    encoder_last_hidden_state_consumed = False
    position_ids = torch.arange(32, dtype=torch.long)
    position_embeds = model.model.decoder.embed_positions.weight[position_ids]
    tokens_to_generate = max_length if test_device.devtype == BackendType.Silicon else 3
    for _ in range(tokens_to_generate):
        if not encoder_last_hidden_state_consumed:
            encoder_last_hidden_state_consumed = True
            tt0.set_active_subgraph(0)
            generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds)
            tt0.push_to_inputs(generate_inputs)
            pybuda.run_generate(input_count=1, write_index=current_token_index//TILE_DIM)
            ans = output_q.get()
        else:
            tt0.set_active_subgraph(1)
            generate_inputs = (decoder_input_ids, decoder_attention_mask, position_embeds)
            tt0.push_to_inputs(generate_inputs)
            pybuda.run_generate(input_count=1, write_index=current_token_index//TILE_DIM)
            ans = output_q.get()

        lm_head_out = ans[0].value().detach()
        scores = logits_processor(decoder_input_ids[:, :current_token_index], lm_head_out[:, current_token_index % TILE_DIM])
        next_token = torch.argmax(scores, dim=-1).item()
        generated_tokens.append(next_token)
        print(f"generated tokens: {tokenizer.decode(generated_tokens)}")

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            position_ids = position_ids + TILE_DIM
            position_embeds = model.model.decoder.embed_positions.weight[position_ids]
            decoder_attention_mask[0, :current_token_index] = 1
            decoder_attention_mask[0, first_current_index:] = 0
            decoder_input_ids[0, :] = tokenizer.pad_token_id

        decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
        decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1
    end = time.time()
    print(f"{len(generated_tokens)} iterations took {end - start} seconds, speed: {(len(generated_tokens)) / (end - start)} iters/sec")
    print(f"generated tokens: {tokenizer.decode(generated_tokens)}")
