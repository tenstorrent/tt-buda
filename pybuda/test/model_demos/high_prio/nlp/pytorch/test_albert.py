# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model

import os

import pybuda
from transformers import AlbertForMaskedLM, AlbertTokenizer, AlbertForTokenClassification, AlbertForSequenceClassification, AlbertForQuestionAnswering

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
def test_albert_masked_lm_pytorch(size, variant, test_device):
    model_ckpt = f"albert-{size}-{variant}"
    
    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AlbertTokenizer.from_pretrained, model_ckpt)
    model = download_model(AlbertForMaskedLM.from_pretrained, model_ckpt)

    pybuda.config.set_configuration_options(
        default_df_override=pybuda.DataFormat.Float16,
        amp_level=2,
    )
    compiler_cfg = pybuda.config._get_global_compiler_config()

    if ("xxlarge" in model_ckpt):
        if test_device.arch == BackendDevice.Grayskull:
            compiler_cfg = pybuda.config._get_global_compiler_config()
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.amp_level = 2
            os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
            if variant == "v2":
                compiler_cfg.enable_enumerate_u_kt = False

        elif test_device.arch == BackendDevice.Wormhole_B0:
            # until tenstorrent/budabackend#1120 is resolved
            pybuda.config.set_configuration_options(
                enable_auto_fusing=False,
                enable_enumerate_u_kt=False,
                amp_level=1,
                default_df_override=pybuda.DataFormat.Float16_b,
            )
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{105*1024}"
            os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "xlarge" == size:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{8*1024}"

        if test_device.arch == BackendDevice.Grayskull:
            os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "large" == size:
        if test_device.arch == BackendDevice.Grayskull:
            compiler_cfg.enable_auto_fusing = False
            os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"
        elif test_device.arch == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"
    elif "base" == size:
        if test_device.arch == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model(**input_tokens)

    verify_module(
        pybuda.PyTorchModule("pt_albertbert_masked_lm", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'])],
        verify_cfg=VerifyConfig(
            enabled=False,
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
   

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
def test_albert_token_classification_pytorch(size, variant, test_device):

    # Set PyBUDA configuration parameters
    pybuda.config.set_configuration_options(
        default_df_override=pybuda.DataFormat.Float16,
        amp_level=2,
    )

    compiler_cfg = pybuda.config._get_global_compiler_config()

    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    # Variants: albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1
    # albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
    model_ckpt = f"albert-{size}-{variant}"
    if "xxlarge" in model_ckpt:
        pybuda.config.set_configuration_options(
            enable_auto_fusing=False,
            enable_enumerate_u_kt=False,
        )
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{105*1024}"
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "xlarge" in model_ckpt:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{8*1024}"

        if test_device.arch == BackendDevice.Grayskull:
            os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "large" == size:
        if test_device.arch == BackendDevice.Grayskull:
            compiler_cfg.enable_auto_fusing = False
            os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"
        elif test_device.arch == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"
    elif "base" == size:
        if test_device.arch == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"


    # Load ALBERT tokenizer and model from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    model = AlbertForTokenClassification.from_pretrained(model_ckpt)

    # Load data sample
    sample_text = "HuggingFace is a company based in Paris and New York"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model(**input_tokens)

    verify_module(
        pybuda.PyTorchModule("pt_albertbert_token_classification", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'])],
        verify_cfg=VerifyConfig(
            enabled=False,
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
