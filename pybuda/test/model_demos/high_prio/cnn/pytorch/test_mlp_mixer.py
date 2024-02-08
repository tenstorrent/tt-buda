# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

import pybuda
import os

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image
from torchvision import transforms
import timm
import requests
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from test.utils import download_model
import urllib
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import sys

varaints = ["mixer_b16_224", "mixer_b16_224_in21k", "mixer_b16_224_miil", "mixer_b16_224_miil_in21k","mixer_b32_224",
            "mixer_l16_224", "mixer_l16_224_in21k", "mixer_l32_224", "mixer_s16_224", "mixer_s32_224"]
@pytest.mark.parametrize("variant", varaints, ids=varaints)
def test_mlp_mixer_timm_pytorch(variant, test_device):
    
    if test_device.arch == BackendDevice.Wormhole_B0 and variant == "mixer_b16_224":
        pytest.skip("Data mismatch PCC=0.80")
        
    if test_device.arch == BackendDevice.Wormhole_B0 and variant == "mixer_b16_224_in21k":
        pytest.skip("Data mismatch PCC=0.68")
        
    if test_device.arch == BackendDevice.Wormhole_B0 \
        and variant in ["mixer_b16_224_miil_in21k", "mixer_l16_224", "mixer_s32_224", "mixer_l16_224_in21k", "mixer_b32_224"]:
        pytest.skip("Python bus error or segfault")
   
    if test_device.arch == BackendDevice.Grayskull \
        and variant in ["mixer_b16_224", "mixer_b16_224_in21k", "mixer_b16_224_miil_in21k", "mixer_l16_224", "mixer_l16_224_in21k", "mixer_s16_224", "mixer_s32_224"]:
            if variant == "mixer_b16_224":
                pytest.skip("Data mismatch on Grayskull PCC = 0.43")
            elif variant == "mixer_b16_224_in21k":
                pytest.skip("Data mismatch on Grayskull PCC = 0.59")
            elif variant == "mixer_b16_224_miil_in21k":
                pytest.skip("Data mismatch on Grayskull PCC = 0.24")
            elif variant == "mixer_l16_224":
                pytest.skip("Data mismatch on Grayskull PCC = 0.68")
            elif variant == "mixer_l16_224_in21k":
                pytest.skip("Bus Error during placer/balancer")
            elif variant == "mixer_s16_224":
                pytest.skip("/home/jenkinsad/pybuda/third_party/budabackend//src/overlay/blob_gen.rb:250:in `ParseStreamString': undefined method `map' for nil:NilClass (NoMethodError)")
            elif variant == "mixer_s32_224":
                pytest.skip("Hangs on Grayskull")

            
   
    model = download_model(timm.create_model, variant, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)


    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"

    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        image = torch.rand(1, 3, 256, 256)
    pixel_values = transform(image).unsqueeze(0)

    # STEP 2: Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(variant+"_pt", model)

    verify_module(
        tt_model,
        input_shapes=[(pixel_values.shape,)],
        inputs=[(pixel_values,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.99,
        ),
    )

