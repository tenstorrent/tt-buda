# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

import os

import pybuda
import torch
import torch.multiprocessing
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms
import urllib
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

torch.multiprocessing.set_sharing_strategy("file_system")
#############

def generate_model_hrnet_imgcls_osmr_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object

    # tenstorrent/pybuda#950
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    
    # STEP 2: Create PyBuda module from PyTorch model
    """
    models = [
        hrnet_w18_small_v1,
        hrnet_w18_small_v2,
        hrnetv2_w18,
        hrnetv2_w30,
        hrnetv2_w32,
        hrnetv2_w40,
        hrnetv2_w44,
        hrnetv2_w48,
        hrnetv2_w64,
    ]
    """
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_hrnet_osmr_{variant}", model)

    # Model load
    os.system(
        "wget -nc https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    torch.hub.download_url_to_file(
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
    )
    input_image = Image.open("dog.jpg")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model
    print(input_batch.shape)
    
    return tt_model, [input_batch], {}


variants = [
    "hrnet_w18_small_v1",
    "hrnet_w18_small_v2",
    "hrnetv2_w18",
    "hrnetv2_w30",
    "hrnetv2_w32",
    "hrnetv2_w40",
    "hrnetv2_w44",
    "hrnetv2_w48",
    "hrnetv2_w64",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_hrnet_osmr_pytorch(variant, test_device):
    model, inputs, _ = generate_model_hrnet_imgcls_osmr_pytorch(
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
            pcc=0.82,
        ),
    )

def generate_model_hrnet_imgcls_timm_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    
    # tenstorrent/pybuda#950
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # STEP 2: Create PyBuda module from PyTorch model
    """
    default_cfgs = {
    'hrnet_w18_small'
    'hrnet_w18_small_v2'
    'hrnet_w18'
    'hrnet_w30'
    'hrnet_w32'
    'hrnet_w40'
    'hrnet_w44'
    'hrnet_w48'
    'hrnet_w64'
    }
    """
    model = download_model(timm.create_model, variant, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_hrnet_timm_{variant}", model)

    ## Preprocessing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    print(input_tensor.shape)
    
    return tt_model, [input_tensor], {}


variants = [
    "hrnet_w18_small",
    "hrnet_w18_small_v2",
    "hrnet_w18",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_hrnet_timm_pytorch(variant, test_device):
    model, inputs, _ = generate_model_hrnet_imgcls_timm_pytorch(
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
            pcc=0.94,
        ),
    )
