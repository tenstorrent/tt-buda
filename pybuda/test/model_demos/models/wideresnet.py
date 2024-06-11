# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
from test.utils import download_model
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
import urllib

def generate_model_wideresnet_imgcls_pytorch(test_device, variant):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # STEP 2: Create PyBuda module from PyTorch model
    framework_model = download_model(torch.hub.load,"pytorch/vision:v0.10.0", variant, pretrained=True)
    framework_model.eval()
    model_name = f"pt_{variant}"
    tt_model = pybuda.PyTorchModule(model_name,framework_model)

    # STEP 3: Prepare input 
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    img_tensor = input_tensor.unsqueeze(0)

    return tt_model, [img_tensor]


def generate_model_wideresnet_imgcls_timm(test_device, variant):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = (pybuda.config._get_global_compiler_config())
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # STEP 2: Create PyBuda module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()
    tt_model = pybuda.PyTorchModule( f"pt_{variant}_timm", framework_model)

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) 
    
    return tt_model, [img_tensor]
