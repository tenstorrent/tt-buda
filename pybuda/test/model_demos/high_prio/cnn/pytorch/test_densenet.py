# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import pybuda
import os 
import urllib.request
from loguru import logger

import skimage
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

import torchxrayvision as xrv

import torch

from PIL import Image
import requests
import urllib
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize, CenterCrop

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

############
def get_input_img():
    try:
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert('RGB')

        transform = Compose([
            Resize(256),
            CenterCrop(224),
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocessing
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)
    print(img_tensor.shape)
    return img_tensor
#############

def get_input_img_hf_xray():
    try:
        img_url = "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
        img_path = "xray.jpg"
        urllib.request.urlretrieve(img_url, img_path)
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        # Add color channel
        img = img[None, :, :]
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 1, 224, 224)
    
    return img_tensor


variants = ["densenet121", "densenet121_hf_xray"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_densenet_121_pytorch(variant, test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test has data mismatch")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    if variant == "densenet121":
        model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        img_tensor = get_input_img()
    else:
        compiler_cfg.enable_tm_cpu_fallback = True
        # Does constant prop on TVM side
        compiler_cfg.enable_tvm_constant_prop = True
        # Fallbacks adv_index to CPU. Used to normalize outputs using threshold extracted as model param (part of output normalization).
        compiler_cfg.cpu_fallback_ops.add("adv_index")
        model_name = "densenet121-res224-all"
        model = download_model(xrv.models.get_model, model_name) 
        img_tensor = get_input_img_hf_xray()


    tt_model = pybuda.PyTorchModule(variant, model)
    
    # STEP 3: Run inference on Tenstorrent device 
    model(img_tensor)

    verify_module(
        tt_model,
        input_shapes=[(img_tensor.shape,)],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def test_densenet_161_pytorch(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with exceeding dram queues error in net2pipe")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"  
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.place_on_new_epoch("concatenate_131.dc.sparse_matmul.7.lc2") 
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

 
    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet161", pretrained=True)
    tt_model = pybuda.PyTorchModule("densnet161_pt", model)
    
    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img() 
    model(img_tensor)

    verify_module(
        tt_model,
        input_shapes=[(img_tensor.shape,)],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
        )
    )
    
    



def test_densenet_169_pytorch(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test has data mismatch")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet169", pretrained=True)
    tt_model = pybuda.PyTorchModule("densnet169_pt", model)
    
    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img() 
    model(img_tensor)

    verify_module(
        tt_model,
        input_shapes=[(img_tensor.shape,)],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )

def test_densenet_201_pytorch(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test has data mismatch")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet201", pretrained=True)
    tt_model = pybuda.PyTorchModule("densnet201_pt", model)
    
    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)

    verify_module(
        tt_model,
        input_shapes=[(img_tensor.shape,)],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
