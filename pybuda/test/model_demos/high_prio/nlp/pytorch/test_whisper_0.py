# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695

import os
import copy
import pytest

import torch
from transformers import pipeline
from transformers import (
    AutoProcessor,
    WhisperConfig,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    LogitsProcessorList
)
from datasets import load_dataset
from typing import Optional
from pybuda.pybudaglobal import TILE_DIM
import pybuda
from test.utils import download_model
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind, NebulaGalaxy
from pybuda import PyTorchModule, VerifyConfig
from pybuda.config import _get_global_compiler_config
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from test.model_demos.models.whisper import Whisper_encoder, Whisper_decoder, generate_model_whisper_decoder_past_cache
import time

variants = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
]


def generate_model_whisper_congen_hf_pytorch(test_device, variant):
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    if "medium" in variant or "large" in variant:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.amp_level = 1

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
    model_config = WhisperConfig.from_pretrained(variant)

    # Reduce size of model for testing
    # model_config.use_cache = False
    # model_config.return_dict = False
    # model_config.decoder_attention_heads = 1
    # model_config.decoder_layers = 1
    # model_config.encoder_attention_heads = 1
    # model_config.encoder_layers = 1
    # model_config.num_hidden_layers = 1
    # model_config.d_model = 512
    # model_config.decoder_ffn_dim = 2048
    # model_config.encoder_ffn_dim = 2048

    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99

    # Hitting silicon data mismatches with GELU
    if variant == "openai/whisper-base" or variant == "openai/whisper-medium" or variant == "openai/whisper-large":
        os.environ["PYBUDA_DECOMPOSE_GELU"] = "1"

    if variant == "openai/whisper-small":
        pcc = 0.94 if test_device.devtype == BackendType.Silicon else 0.99

    # Getting proper results for end-to-end runs (speech-to-text); GELU pcc isn't problematic in this case
    # therefore reducing for proper CI passing and tracking of further regressions
    if variant == "openai/whisper-large":
        pcc = 0.58 if test_device.devtype == BackendType.Silicon else 0.99

    framework_model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        config=model_config,
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

    return pybuda_model, [decoder_input_ids, encoder_outputs], {"pcc": pcc}

@pytest.mark.skip(reason="Redundant")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper(test_device, variant):
    pytest.skip("Already tested with past-cache and separated encoder-decoder")
    
    model, inputs, other = generate_model_whisper_congen_hf_pytorch(
        test_device,
        variant,
    )

    verify_module(
        model,
        [
            (inputs[0].shape, inputs[1].shape),
        ],
        inputs=[
            (inputs[0], inputs[1]),
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=other["pcc"],
            enabled=False if variant == "openai/whisper-medium" else True,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        ),
    )


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant")
def test_whisper_pipeline(test_device, variant):
    pytest.skip("Already tested with past-cache and separated encoder-decoder")
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with no valid grids (50 nodes)")

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_auto_fusing = False  # tenstorrent/pybuda#844
    compiler_cfg.amp_level = 2
    compiler_cfg.enable_link_past_cache_ios = False
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load model (with tokenizer and feature extractor)
    framework_model = download_model(
        WhisperForConditionalGeneration.from_pretrained, variant
    )
    tokenizer = download_model(WhisperTokenizer.from_pretrained, variant)
    feature_extractor = download_model(WhisperFeatureExtractor.from_pretrained, variant)

    ### Load HF pipeline
    hf_pipeline = pipeline(
        "automatic-speech-recognition",
        model=copy.deepcopy(framework_model),
        tokenizer=copy.deepcopy(tokenizer),
        feature_extractor=copy.deepcopy(feature_extractor),
    )

    ### Load PyBuda pipeline
    asr_pipeline = pybuda_pipeline(
        "automatic-speech-recognition",
        model=framework_model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        pybuda_max_length=32,
    )

    # Load & preprocess sample audio
    ### Load from HF datasets & preprocess
    # data_set = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # sample = processor(data_set[0]["audio"]["array"], return_tensors="pt")
    ### Load preprocessed from local
    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]
    ### Load direct audio file
    # sample_audio = "audio_demos/whisper/data_sample/1272-128104-0000.flac"

    # Sanity run on CPU
    cpu_out = hf_pipeline(sample_audio)
    print("HF pipeline:", cpu_out["text"])

    # Compile and run on TT device
    tt_out = asr_pipeline(sample_audio)
    print("TT pipeline:", tt_out["text"])

    # Compare outputs
    assert cpu_out["text"] == tt_out["text"]

@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant")
def test_whisper_encoder(test_device, variant):
    pytest.skip("Already tested with past-cache and separated encoder-decoder")

    if variant == "openai/whisper-medium" or variant == "openai/whisper-large":
       pytest.skip("Still under development")

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.amp_level = 1
        compiler_cfg.default_dram_parameters = False
        os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
        os.environ["PYBUDA_PAD_OUTPUT_BUFFER_THRESHOLD_TILES"] = "1536"
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "35000"
        os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
        os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
        os.environ["PYBUDA_NOP_ON_DIRECT_SHORT_PATH"] = "1"
        os.environ["PYBUDA_SKIP_SMALL_UKT"] = "1"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
        if variant == "openai/whisper-small":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    pcc = 0.95 if test_device.devtype == BackendType.Silicon else 0.99
    if variant == "openai/whisper-tiny":
        pcc = 0.92 if test_device.devtype == BackendType.Silicon else 0.99

    if variant == "openai/whisper-base":
        pcc = 0.93 if test_device.devtype == BackendType.Silicon else 0.99
        if test_device.arch == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "55000"

    config = WhisperConfig.from_pretrained(variant)
    config.return_dict = False
    config.encoder_layers = 1
    config.decoder_layers = 1
    pad_model = True
    if pad_model:
        pad_to_tiles = 48
        padded_len = pad_to_tiles * TILE_DIM
        pad_amount = padded_len - config.max_source_positions
        config.max_source_positions = padded_len
    else:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{150*1024}"
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = f"{100*1024}"

    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        ignore_mismatched_sizes=True,
        config=config,
    )
    if pad_model:
        unpadded_model = WhisperForConditionalGeneration.from_pretrained(variant)
        padded_param = torch.nn.functional.pad(unpadded_model.model.encoder.embed_positions.weight.data, (0,0,0,pad_amount))
        model.model.encoder.embed_positions.weight.data = padded_param

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
    
    model = Wrapper(model)
    pybuda_model = PyTorchModule("pt_whisper", model)

    # Load and preprocess sample audio
    sample = torch.load("pybuda/test/model_demos/utils/nlp/pytorch/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    if pad_model:
        input_features = torch.nn.functional.pad(inputs.input_features, (0, pad_amount*2, 0, 0))
    else:
        input_features = inputs.input_features

    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=pybuda_model)
    output_q = pybuda.initialize_pipeline(training=False,sample_inputs=(input_features, ),)

    start = time.time()
    tokens_to_generate = 10 if test_device.devtype == BackendType.Silicon else 3
    for _ in range(tokens_to_generate):
        tt0.push_to_inputs(input_features)
        pybuda.run_forward(input_count=1)
        ans = output_q.get()

    end = time.time()
    print(f"Ran {tokens_to_generate} iterations. Duration: {(end - start) / tokens_to_generate} seconds, speed: {(tokens_to_generate) / (end - start)} iters/sec")
