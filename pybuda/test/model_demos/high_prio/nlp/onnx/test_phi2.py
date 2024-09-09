# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
import pytest
import os
import onnx
from transformers import AutoTokenizer
from pybuda._C.backend_api import BackendDevice

# Masked fill kernal produced invalid results in Silicon BackendType 
# Masked fill is converted to the Where operation after exporting the model to ONNX.
# The Where operation also produces invalid results on the Silicon BackendType, similar to Masked fill.
# So Disabling the verification in BBE for Silicon BackendType for causal LM task
# Issue link - https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2837

variants = ["microsoft/phi-2", "microsoft/phi-2-pytdml"]


@pytest.mark.parametrize("variant", variants)
def test_phi2_onnx(variant, test_device):

    # pybuda Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "20480"

    elif test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_DRAM_PICK_CAPACITY"] = "1"
        os.environ["PYBUDA_DRAM_FLIP_FLOP"] = "1"

    # load the Tokenizer
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

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    variant_name = str(variant.split("/")[-1].replace("-", "_"))
    model_name = f"onnx_{variant_name}"
    load_path = f"third_party/confidential_customer_models/internal/phi2/files/onnx/{variant_name}/decoder_model.onnx"

    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule(model_name, model, load_path)

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
