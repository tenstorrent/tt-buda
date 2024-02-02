# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, DataFormat

import os

import pybuda
import requests
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

def generate_model_vit_imgcls_hf_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # STEP 2: Create PyBuda module from PyTorch model
    image_processor = download_model(AutoImageProcessor.from_pretrained,
        variant
    )
    model = download_model(ViTForImageClassification.from_pretrained,
        variant
    )
    tt_model = pybuda.PyTorchModule("ViT_classif_16_224", model)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits
    
    return tt_model, [img_tensor], {}


dataset = load_dataset("huggingface/cats-image")
image_1 = dataset["test"]["image"][0]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_2 = Image.open(requests.get(url, stream=True).raw)

variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classify_224_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device, variant,
    )

    if "PYBUDA_NEB_GALAXY_CI" in os.environ:
        chip_ids = [0, 11, 10, 9, 8, 7, 19, 20, 21, 22, 23, 24, 6, 5, 14, 13, 12, 16, 15, 3, 4, 26, 25, 32, 31, 30, 29, 28, 27, 1, 2, 18, 17]
    else:
        chip_ids = [0]

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            chip_ids=chip_ids
        )
    )

variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classify_224_hf_pytorch_1x1(variant, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    if "large" in variant:
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "20000"


    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device, variant,
    )

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9
        )
    )
