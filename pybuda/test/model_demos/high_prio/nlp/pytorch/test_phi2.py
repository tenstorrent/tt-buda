# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice
import torch
from transformers import PhiForCausalLM, AutoTokenizer, PhiConfig, PhiForTokenClassification, PhiForSequenceClassification
import os
import pytest

# Masked fill kernal produced invalid results in Silicon BackendType
# So Disabling the verification in BBE for Silicon BackendType for causal LM task
# Issue link - https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2712

variants = ["microsoft/phi-2", "microsoft/phi-2-pytdml"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_phi2_clm(test_device, variant):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "20480"
    compiler_cfg.amp_level = 1
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    if test_device.arch == BackendDevice.Blackhole:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{32*1024}"

    # Load PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load model and tokenizer from HuggingFace
    model = PhiForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # input_prompt
    input_prompt = "Write a detailed analogy between mathematics and a lighthouse."

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(torch.int32)
    attn_mask = inputs["attention_mask"].to(torch.float32)

    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape,attn_mask.shape,)],
        inputs=[(input_ids,attn_mask,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False if test_device.devtype == pybuda.BackendType.Silicon else True,
        ),
    )


@pytest.mark.parametrize("variant", variants)
def test_phi2_token_classification(test_device, variant):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = PhiForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]

    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    pcc = 0.99

    if test_device.devtype == pybuda.BackendType.Silicon and test_device.arch == BackendDevice.Wormhole_B0:
        pcc = 0.98
    elif test_device.devtype == pybuda.BackendType.Silicon and test_device.arch == BackendDevice.Grayskull:
        pcc = 0.97

    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape,)],
        inputs=[(input_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )


@pytest.mark.parametrize("variant", variants)
def test_phi2_sequence_classification(test_device, variant):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = PhiForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "I am not satisfied with the quality of this product."

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]

    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape,)],
        inputs=[(input_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
