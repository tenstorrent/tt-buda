# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

import os

import torch
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
    GPTNeoConfig,
    GPTNeoForSequenceClassification,
)


variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gptneo_causal_lm(variant, test_device):
    # Set random seed for repeatability
    torch.manual_seed(42)

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config() 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    compiler_cfg.balancer_policy = "Ribbon"

    if test_device.arch == BackendDevice.Wormhole_B0:
        if variant == "EleutherAI/gpt-neo-2.7B":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
        if variant == "EleutherAI/gpt-neo-1.3B":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "76444"

    elif test_device.arch == BackendDevice.Blackhole:
        if variant == "EleutherAI/gpt-neo-125M":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{12*1024}"

    # Load tokenizer and model
    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B,
    # EleutherAI/gpt-neo-2.7B

    config = download_model(GPTNeoConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPTNeoConfig(**config_dict)

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(GPTNeoForCausalLM.from_pretrained, variant, config=config)

    # Sample input text
    prompt = "My name is Bert, and I am"

    inputs = tokenizer(prompt, return_tensors="pt",max_length=256, pad_to_max_length=True, truncation=True)

    # Wrapper to get around attention mask
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    if "PYBUDA_NEB_GALAXY_CI" in os.environ:
        chip_ids = [0, 11, 10, 9, 8, 7, 19, 20, 21, 22, 23, 24, 6, 5, 14, 13, 12, 16, 15, 3, 4, 26, 25, 32, 31, 30, 29, 28, 27, 1, 2, 18, 17]
    else:
        chip_ids = [0]

    tt_model = pybuda.PyTorchModule("gptneo_generation", Wrapper(model))
    verify_module(
        tt_model,
        input_shapes=[
            (
                input_ids.shape,
                attn_mask.shape,
            )
        ],
        inputs=[
            (
                input_ids,
                attn_mask,
            )
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False,
            chip_ids=chip_ids
        ),
    )


variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gptneo_sequence_classification(variant, test_device):
    # Load tokenizer and model from HuggingFace
    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B,
    # EleutherAI/gpt-neo-2.7B
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    if variant in ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(
        GPTNeoForSequenceClassification.from_pretrained, variant, torchscript=True
    )

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    verify_module(
        pybuda.PyTorchModule("pt_gptneo_seq_classification", Wrapper(model)),
        input_shapes=[
            (
                input_tokens["input_ids"].shape,
                input_tokens["attention_mask"].shape,
            )
        ],
        inputs=[
            (
                input_tokens["input_ids"],
                input_tokens["attention_mask"],
            )
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False,
        ),
    )
