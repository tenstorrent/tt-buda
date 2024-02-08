# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

import pybuda
import os 

from pytorchcv.model_provider import get_model as ptcv_get_model

import torch
from PIL import Image
from torchvision import transforms
from vgg_pytorch import VGG 
from loguru import logger
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import urllib
from torchvision import transforms


variants = ["vgg11", "vgg13", "vgg16", "vgg19", "bn_vgg19", "bn_vgg19b"]
@pytest.mark.parametrize("variant", variants)
def test_vgg_osmr_pytorch(variant, test_device):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    if (test_device.arch == BackendDevice.Wormhole_B0):
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    # Variants: 
    #['vgg11', 'vgg13', 'vgg16', 'vgg19', 
    # 'bn_vgg11', 'bn_vgg13', 'bn_vgg16', 'bn_vgg19',
    # 'bn_vgg11b', 'bn_vgg13b', 'bn_vgg16b', 'bn_vgg19b']
    #model = src_VGG_Osmr.vgg11(pretrained=True)
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_{variant}_osmr", model)

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        input_batch = torch.rand(1, 3, 224, 224)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            enabled=False,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def test_vgg_19_hf_pytorch(test_device):
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    '''
    # https://pypi.org/project/vgg-pytorch/
    # Variants:
    vgg11, vgg11_bn
    vgg13, vgg13_bn	    
    vgg16, vgg16_bn
    vgg19, vgg19_bn
    ''' 
    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(VGG.from_pretrained, 'vgg19')
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_vgg_19_hf", model)

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        input_batch = torch.rand(1, 3, 224, 224)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
  
def preprocess_timm_model(model_name):
   model = timm.create_model(model_name, pretrained=True)
   model.eval()
   try:
       config = resolve_data_config({}, model=model)
       transform = create_transform(**config)
       url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
       urllib.request.urlretrieve(url, filename)
       img = Image.open(filename).convert('RGB')
       img_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
   except:
       logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
       img_tensor = torch.rand(1, 3, 224, 224) 
   
   return model, img_tensor

def test_vgg_bn19_timm_pytorch(test_device):
    torch.multiprocessing.set_sharing_strategy("file_system") 
    model_name = 'vgg19_bn'
    model, image_tensor = download_model(preprocess_timm_model, model_name)

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    # STEP 2: Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(model_name+"_timm_pt", model)

    # STEP 3: Run inference on Tenstorrent device 
    verify_module(
        tt_model,
        input_shapes=[(image_tensor.shape,)],
        inputs=[(image_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            pcc=0.9
        )
    )

def test_vgg_bn19_torchhub_pytorch(test_device):
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    
    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
 
    # STEP 2: Create PyBuda module from PyTorch model
    # Variants:
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    model = download_model(torch.hub.load, 'pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_vgg_bn19_torchhub", model)

 
    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        input_batch = torch.rand(1, 3, 224, 224)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            pcc = 0.98 if test_device.arch == BackendDevice.Grayskull else 0.99
        )
    )
