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
import requests
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import urllib
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger

def generate_model_resnet_imgcls_hf_pytorch(test_device, variant):    
    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    feature_extractor = download_model(AutoFeatureExtractor.from_pretrained, model_ckpt)
    model = download_model(ResNetForImageClassification.from_pretrained, model_ckpt)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for output queue (perf)
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"

    # Load data sample
    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg" 
        image = Image.open(requests.get(url, stream=True).raw)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        image = torch.rand(1, 3, 256, 256)

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    model = pybuda.PyTorchModule("pt_resnet50", model)
    
    return model, [pixel_values], {}


@pytest.mark.parametrize("enable_default_dram_parameters", [True, False])
def test_resnet(test_device, enable_default_dram_parameters):
    if test_device.arch == BackendDevice.Grayskull and enable_default_dram_parameters == False:
        pytest.skip("Failing on GS with: Core (c=0,y=8,x=1) [routing]  (c=0,y=6,x=0) [worker] [op_name=add_69] exceeded resource constraints: active dram queues used: 56 limit: 40")
    
    model, inputs, _ = generate_model_resnet_imgcls_hf_pytorch(
        test_device, "microsoft/resnet-50",
    )

    compiler_cfg = pybuda.config._get_global_compiler_config()

    verify_module(
        model,
        input_shapes=[inputs[0].shape],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )

    
def generate_model_resnet_imgcls_timm_pytorch(test_device, variant):
    # Load ResNet50 feature extractor and model from TIMM
    model = download_model(timm.create_model, variant, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        image = torch.rand(1, 3, 256, 256)

    # Data preprocessing
    pixel_values = transform(image).unsqueeze(0) 
    
    model = pybuda.PyTorchModule("pt_resnet50", model)

    return model, [pixel_values], {}


def test_resnet_timm(test_device):
    model, inputs, _ = generate_model_resnet_imgcls_timm_pytorch(
        test_device, "resnet50",
    )

    verify_module(
        model,
        input_shapes=[inputs[0].shape],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
