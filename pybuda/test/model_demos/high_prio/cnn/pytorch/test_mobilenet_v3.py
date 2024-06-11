# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy


import os
import urllib
import pybuda
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger

def generate_model_mobilenetV3_imgcls_torchhub_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    model = download_model(torch.hub.load,
        "pytorch/vision:v0.10.0", variant, pretrained=True
    )
    tt_model = pybuda.PyTorchModule("mobilenet_v3_large_pt", model)

    # Run inference on Tenstorrent device
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision, to make a compatible postprocessing of the predicted class
    preprocessor = AutoImageProcessor.from_pretrained(
        "google/mobilenet_v2_1.0_224"
    )
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values
    
    return tt_model, [image_tensor], {}


variants = ["mobilenet_v3_large", "mobilenet_v3_small"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mobilenetv3_basic(variant, test_device):
    model, inputs, _ = generate_model_mobilenetV3_imgcls_torchhub_pytorch(
        test_device, variant,
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
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            enabled=False # TODO: small variant has very low PCC, large variant has high PCC
        )
    )

def generate_model_mobilenetV3_imgcls_timm_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    # Both options are good
    # model = timm.create_model('mobilenetv3_small_100', pretrained=True)
    if variant == "mobilenetv3_small_100":
        model = download_model(timm.create_model,
            f"hf_hub:timm/mobilenetv3_small_100.lamb_in1k", pretrained=True
        )
    else:
        model = download_model(timm.create_model,
            f"hf_hub:timm/mobilenetv3_large_100.ra_in1k", pretrained=True
        )
        
    tt_model = pybuda.PyTorchModule(variant, model)

    # Image load and pre-processing into pixel_values
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        image_tensor = transform(img).unsqueeze(
            0
        )  # transform and add batch dimension
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        image_tensor = torch.rand(1, 3, 224, 224)

    
    return tt_model, [image_tensor], {}



variants = ["mobilenetv3_large_100", "mobilenetv3_small_100"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mobilenetv3_timm(variant, test_device):
    model, inputs, _ = generate_model_mobilenetV3_imgcls_timm_pytorch(
        test_device, variant,
    )

    os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    verify_module(
        model,
        input_shapes=[inputs[0].shape],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            enabled=False # TODO: small variant has very low PCC, large variant has high PCC
        )
    )

variants = ["mobilenetv3_large_100", "mobilenetv3_small_100"]
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Not supported")
def test_mobilenetv3_timm_1x1(variant, test_device):
    pytest.skip()
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    model, inputs, _ = generate_model_mobilenetV3_imgcls_timm_pytorch(
        test_device, variant,
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
            enabled=False # TODO: small variant has very low PCC, large variant has high PCC
        )
    )
