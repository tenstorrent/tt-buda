# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, DataFormat, NebulaGalaxy

import os

import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, XGLMForCausalLM, XGLMConfig


variants = ["facebook/xglm-564M", "facebook/xglm-1.7B"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xglm_causal_lm(variant, test_device):
    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("take")
    compiler_cfg.default_df_override = DataFormat.Float16_b
    compiler_cfg.enable_enumerate_u_kt = False
    if variant == "facebook/xglm-1.7B":
        compiler_cfg.amp_level = 1
        if test_device.arch == BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{16*1024}"
    if (test_device.arch == BackendDevice.Grayskull and variant == "facebook/xglm-564M") or (test_device.arch == BackendDevice.Wormhole_B0):
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/xglm-564M", "facebook/xglm-1.7B"

    config = XGLMConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict['return_dict'] = False
    config_dict['use_cache'] = False
    config = XGLMConfig(**config_dict)
    model = download_model(XGLMForCausalLM.from_pretrained, variant, config=config)
    model.eval()
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"
    input_tokens = tokenizer(
        prefix_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )   

    pcc_value = 0.99
    if test_device.arch == BackendDevice.Wormhole_B0 and test_device.devtype == BackendType.Silicon:
        if variant == "facebook/xglm-564M":
            pcc_value = 0.91
        elif variant == "facebook/xglm-1.7B":
            pcc_value = 0.94
    elif test_device.arch == BackendDevice.Grayskull and test_device.devtype == BackendType.Silicon:
        if variant == "facebook/xglm-564M":
            pcc_value = 0.89
        elif variant == "facebook/xglm-1.7B":
            pcc_value = 0.98

             
    verify_module(
        pybuda.PyTorchModule("pt_xglm_causal_lm", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            pcc=pcc_value,
        )
    )
