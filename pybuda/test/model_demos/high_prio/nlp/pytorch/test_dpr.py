# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

import os

import pybuda
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRReader, DPRReaderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

variants = ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_context_encoder_pytorch(variant, test_device):

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-ctx_encoder-single-nq-base, facebook/dpr-ctx_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRContextEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRContextEncoder.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = "Hello, is my dog cute?"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_dpr_context_encoder", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape, input_tokens['token_type_ids'].shape)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'], input_tokens['token_type_ids'])],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.98,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
variants = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_question_encoder_pytorch(variant, test_device):
    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-question_encoder-single-nq-base, facebook/dpr-question_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRQuestionEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRQuestionEncoder.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = "Hello, is my dog cute?"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_dpr_question_encoder", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape, input_tokens['token_type_ids'].shape)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'], input_tokens['token_type_ids'])],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )

variants = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(variant, test_device):
    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRReaderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRReader.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"
 
    # Data preprocessing
    input_tokens = tokenizer(
        questions=["What is love?"],
        titles=["Haddaway"],
        texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_dpr_reader", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'])],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
