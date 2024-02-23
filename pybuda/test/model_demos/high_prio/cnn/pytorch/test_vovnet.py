# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
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
import urllib
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger

import sys
 
 
def get_image():
    try:
        torch.hub.download_url_to_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
        )
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = preprocess(input_image)
        img_tensor = img_tensor.unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)

    return img_tensor


def generate_model_vovnet_imgcls_osmr_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # STEP 2: Create PyBuda module from PyTorch model 
    model = download_model(ptcv_get_model, variant, pretrained=True)
    tt_model = pybuda.PyTorchModule(f"{variant}_osmr_pt", model) 
    
    image_tensor = get_image()
    
    return tt_model, [image_tensor], {}


varaints = ["vovnet27s", "vovnet39", "vovnet57"]
@pytest.mark.parametrize("variant", varaints, ids=varaints)
def test_vovnet_osmr_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vovnet_imgcls_osmr_pytorch(
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
            pcc=0.85
        )
    )


#https://github.com/stigma0617/VoVNet.pytorch
sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
from src_vovnet_stigma import vovnet39  
 
def preprocess_steps(model_type):
    model = model_type(True, True).eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    try:
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert('RGB')
        img_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor


def generate_model_vovnet39_imgcls_stigma_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
   
    # STEP 2: Create PyBuda module from PyTorch model 
    model, image_tensor = download_model(preprocess_steps, vovnet39)
    tt_model = pybuda.PyTorchModule("vovnet_39_stigma_pt", model) 
    
    return tt_model, [image_tensor], {}

   
@pytest.mark.parametrize("enable_default_dram_parameters", [True, False])
def test_vovnet_v1_39_stigma_pytorch(test_device, enable_default_dram_parameters):
    model, inputs, _ = generate_model_vovnet39_imgcls_stigma_pytorch(
        test_device, None,
    )

    if enable_default_dram_parameters == True:
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_dram_parameters = enable_default_dram_parameters

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
    
from src_vovnet_stigma import vovnet57   


def generate_model_vovnet57_imgcls_stigma_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
   
    # STEP 2: Create PyBuda module from PyTorch model 
    model, image_tensor = download_model(preprocess_steps, vovnet57)
    tt_model = pybuda.PyTorchModule("vovnet_57_stigma_pt", model) 
    
    return tt_model, [image_tensor], {}



   
def test_vovnet_v1_57_stigma_pytorch(test_device):
    model, inputs, _ = generate_model_vovnet57_imgcls_stigma_pytorch(
        test_device, None,
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
            pcc=0.95
        )
    )

def preprocess_timm_model(model_name):
   model = timm.create_model(model_name, pretrained=True)
   model.eval()
   config = resolve_data_config({}, model=model)
   transform = create_transform(**config)

   try:
       url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
       urllib.request.urlretrieve(url, filename)
       img = Image.open(filename).convert('RGB')
       img_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
   except:
       logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
       img_tensor = torch.rand(1, 3, 224, 224)
   
   return model, img_tensor


def generate_model_vovnet_imgcls_timm_pytorch(test_device, variant):
    model, image_tensor = download_model(preprocess_timm_model, variant)
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    
    # tenstorrent/pybuda#915
    if test_device.arch == BackendDevice.Grayskull and variant == "ese_vovnet39b":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(variant+"_pt", model)
    
    return tt_model, [image_tensor], {}


variants = ["ese_vovnet19b_dw", "ese_vovnet39b", "ese_vovnet99b"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vovnet_timm_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vovnet_imgcls_timm_pytorch(
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
            pcc=0.95
        )
    )
