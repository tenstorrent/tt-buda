# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# CodeGen Demo - CasualLM

import os
import torch
import pytest
from test.utils import download_model
from transformers import AutoTokenizer, CodeGenForCausalLM

import pybuda
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind, NebulaGalaxy
from pybuda.verify.backend import verify_module
from pybuda._C.backend_api import BackendDevice, BackendType

variants = [
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-nl",
]

@pytest.mark.parametrize("variant", variants, ids=variants)
def test_codegen(test_device, variant):
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON2"] = "1"
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.balancer_op_override("matmul_1829", "grid_shape", (2, 8))

    pcc_value = 0.99
    if test_device.arch == BackendDevice.Wormhole_B0:
        if test_device.devtype == BackendType.Silicon:
            if variant == "Salesforce/codegen-350M-multi":
                pcc_value = 0.96
            elif variant == "Salesforce/codegen-350M-nl":
                pcc_value = 0.95
    elif test_device.arch == BackendDevice.Grayskull:
        if test_device.devtype == BackendType.Silicon:
            if variant == "Salesforce/codegen-350M-mono":
                pcc_value = 0.96
            elif variant == "Salesforce/codegen-350M-multi":
                pcc_value = 0.93
            elif variant == "Salesforce/codegen-350M-nl":
                compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16
                pcc_value = 0.90

    # Load model (with tokenizer)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    framework_model = download_model(
        CodeGenForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False
    )

    # Input prompt
    input_prompt = "def hello_world():"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    # Wrapper to get around attention mask
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    framework_model = Wrapper(framework_model)

    # Sanity run
    input_ids = input_ids.to(torch.int32)
    attn_mask = attn_mask.to(torch.float32)
    out = framework_model(input_ids, attn_mask)

    pybuda_model = pybuda.PyTorchModule("pt_"+str(variant.split("/")[-1].replace("-", "_")), framework_model)
    verify_module(
        pybuda_model,
        input_shapes=[(input_ids.shape, attn_mask.shape,)],
        inputs=[(input_ids, attn_mask,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            pcc=pcc_value,
        ),
    )
