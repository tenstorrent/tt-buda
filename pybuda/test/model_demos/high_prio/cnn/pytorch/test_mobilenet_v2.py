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
import urllib
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
from loguru import logger
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import MobileNetV2FeatureExtractor, MobileNetV2ForSemanticSegmentation

def generate_model_mobilenetV2_imgcls_torchhub_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.balancer_policy = "CNN"

    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(torch.hub.load,
        variant, "mobilenet_v2", pretrained=True
    )
    tt_model = pybuda.PyTorchModule("mobilenet_v2", model)

    # Image preprocessing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision,
    # to make a compatible postprocessing of the predicted class
    preprocessor = download_model(AutoImageProcessor.from_pretrained,
        "google/mobilenet_v2_1.0_224"
    )
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values
    
    return tt_model, [image_tensor], {}

def test_mobilenetv2_basic(test_device):
    model, inputs, _ = generate_model_mobilenetV2_imgcls_torchhub_pytorch(
        test_device, "pytorch/vision:v0.10.0",
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
            pcc=0.95,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def generate_model_mobilenetV2I96_imgcls_hf_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained,
        variant
    )
    model = download_model(AutoModelForImageClassification.from_pretrained,
        variant
    )
    tt_model = pybuda.PyTorchModule("mobilenet_v2__hf_035_96", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values
    
    return tt_model, [image_tensor], {}

    
def test_mobilenetv2_96(test_device):
    model, inputs, _ = generate_model_mobilenetV2I96_imgcls_hf_pytorch(
        test_device, "google/mobilenet_v2_0.35_96",
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
            pcc=0.8,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )

def generate_model_mobilenetV2I160_imgcls_hf_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained,
        variant
    )
    model = download_model(AutoModelForImageClassification.from_pretrained,
        variant
    )
    tt_model = pybuda.PyTorchModule("mobilenet_v2__hf_075_160", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values
    
    return tt_model, [image_tensor], {}


def test_mobilenetv2_160(test_device):
    model, inputs, _ = generate_model_mobilenetV2I160_imgcls_hf_pytorch(
        test_device, "google/mobilenet_v2_0.75_160",
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

def generate_model_mobilenetV2I244_imgcls_hf_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_RIBBON2"] = "1"

    # Create PyBuda module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained,
        variant
    )
    model = download_model(AutoModelForImageClassification.from_pretrained,
        variant
    )
    tt_model = pybuda.PyTorchModule("mobilenet_v2__hf_1_224", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    
    image_tensor = inputs.pixel_values
    
    return tt_model, [image_tensor], {}

    
def test_mobilenetv2_224(test_device):
    model, inputs, _ = generate_model_mobilenetV2I244_imgcls_hf_pytorch(
        test_device, "google/mobilenet_v2_1.0_224",
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
    
def generate_model_mobilenetV2_imgcls_timm_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_RIBBON2"] = "1"

    # Create PyBuda module from PyTorch model
    model = download_model(timm.create_model, variant, pretrained=True)
    tt_model = pybuda.PyTorchModule("mobilenet_v2__hf_timm", model)

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

    
def test_mobilenetv2_timm(test_device):
    model, inputs, _ = generate_model_mobilenetV2_imgcls_timm_pytorch(
        test_device, "mobilenetv2_100",
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
            pcc=0.95,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )


def generate_model_mobilenetV2_semseg_hf_pytorch(test_device, variant):
    # This variant with input size 3x224x224 works with manual kernel fracturing
    # of the first op. Pad between input activations and first convolution needs
    # to be hoist to the input in order for pre-striding to work (no need for
    # manual kernel fracturing).
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Failing on GS with: Could not reconcile constraints: path[conv2d_554.dc.matmul.8 -> add_567]")
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Load model
    framework_model = download_model(MobileNetV2ForSemanticSegmentation.from_pretrained, variant)
    pybuda_model = pybuda.PyTorchModule("pt_mobilenet_v2_deeplab_v3", framework_model)

    # I 3x513x513
    # # Load and pre-process image
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image_processor = AutoImageProcessor.from_pretrained(variant)
    # img_tensor = image_processor(images=image, return_tensors="pt").pixel_values
    
    # II 3x224x224
    # Load and pre-process image
    try:
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)

    # Sanity run
    # cpu_out = framework_model(img_tensor)
    # cpu_predicted_mask = cpu_out.logits.argmax(1).squeeze(0) 
    # print("Predicted mask", cpu_predicted_mask)  
    
    return pybuda_model, [img_tensor], {}


variants = [
    "google/deeplabv3_mobilenet_v2_1.0_513"
]

@pytest.mark.parametrize("variant", variants)
def test_mobilenetv2_deeplabv3(variant, test_device):
    model, inputs, _ = generate_model_mobilenetV2_semseg_hf_pytorch(
        test_device, variant,
    )

    # Verify
    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.85,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        ),
    )
