# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import os
import urllib
from test.utils import download_model
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize, CenterCrop
import requests
from loguru import logger

from PIL import Image
import numpy as np
import pytest

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind
from pytorchcv.model_provider import get_model as ptcv_get_model
import segmentation_models_pytorch as smp 
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def generate_model_unet_imgseg_osmr_pytorch(test_device, variant):
    # Also, golden test segfaults when pushing params to golden: tenstorrent/pybuda#637
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_FORCE_RESIZE_DENSE_MM"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"
    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.balancer_policy = "CNN"

    # STEP 2: Create PyBuda module from PyTorch model 
    model = download_model(ptcv_get_model, variant, pretrained=False)
    tt_model = pybuda.PyTorchModule("unet_cityscapes_osmr_pt", model) 

    # STEP 3: Run inference on Tenstorrent device 
    img_tensor = x = torch.randn(1, 3, 224, 224)
    #img_tensor = x = torch.randn(1, 3, 1024, 2048)
    #output = model(img_tensor)
    
    return tt_model, [img_tensor], {}

    
def test_unet_osmr_cityscape_pytorch(test_device):
    model, inputs, _ = generate_model_unet_imgseg_osmr_pytorch(
        test_device, "unet_cityscapes",
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
            pcc = 0.98 if test_device.arch == BackendDevice.Grayskull else 0.99 # Grayskull PCC is about 0.986
        )
    )


def get_imagenet_sample():
    try:
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert('RGB')

        # Preprocessing
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
    return img_tensor


@pytest.mark.skip(reason="Not supported")
def test_unet_holocron_pytorch(test_device):
    from holocron.models.segmentation.unet import unet_tvvgg11 
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(unet_tvvgg11, pretrained= True).eval()
    tt_model = pybuda.PyTorchModule("unet_holocron_pt", model) 

    img_tensor = get_imagenet_sample()

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

    # STEP 3: Run inference on Tenstorrent device
    #output = model(img_tensor)
    # output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    # output = output_q.get()[0].value()
   
    # #print(output)
    # prob_mask = output.sigmoid()
    # #print(prob_mask)
    # pred_mask = (prob_mask > 0.5).float()
    # print(pred_mask)
    #print()
    #print(mask)
    #print(pred_mask.mean(), "--", mask.mean())
    
    #File "/home/mbahnas/GitLab/PYBUDA/pybuda_0317/pybuda/pybuda/pybuda/op/eval/pybuda/resize.py", line 61, in shape
    #AssertionError: Only support upsample with integer scale factor


def generate_model_unet_imgseg_smp_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"]= "FastCut"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1488"] = 3

    # STEP 2: Create PyBuda module from PyTorch model
    #encoder_name = "vgg19"
    encoder_name = "resnet101"
    #encoder_name = "vgg19_bn"
     
    model = download_model(smp.Unet, 
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    model.eval()
    
    tt_model = pybuda.PyTorchModule("unet_qubvel_pt", model) 

    # Image preprocessing
    params = download_model(smp.encoders.get_preprocessing_params, encoder_name)
    std = torch.tensor(params["std"]).view(1, 3, 1, 1)
    mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
    
    image = get_imagenet_sample()
    img_tensor = torch.tensor(image)
    img_tensor = (img_tensor - mean)/std
    print(img_tensor.shape)

    return tt_model, [img_tensor], {}

 
def test_unet_qubvel_pytorch(test_device):
    model, inputs, _ = generate_model_unet_imgseg_smp_pytorch(
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
            pcc = 0.92 if test_device.arch == BackendDevice.Grayskull else 0.97 # Grayskull PCC is about 0.925, WH B0 PCC is about 0.976
        )
    )
    
def generate_model_unet_imgseg_torchhub_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.balancer_op_override("conv2d_transpose_174.dc.conv2d.17.dc.matmul.11", "grid_shape", (4,4))


    # STEP 2: Create PyBuda module from PyTorch model
    model = download_model(torch.hub.load,
            "mateuszbuda/brain-segmentation-pytorch",
            variant,
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_unet_torchhub", model)
    
    # Download an example input image  
    url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(input_image)
    img_batch = input_tensor.unsqueeze(0)
    
    return tt_model, [img_batch], {}



def test_unet_torchhub_pytorch(test_device):
    pybuda.config.override_op_size("_fused_op_6", (2, 2))

    model, inputs, _ = generate_model_unet_imgseg_torchhub_pytorch(
        test_device, "unet",
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
            enabled=False,
        )
    )
    # print(img_batch.shape)
 
    
    # # STEP 3: Run inference on Tenstorrent device 
    # #output = model(img_batch)
    # output_q = pybuda.run_inference(tt_model, inputs=([img_batch]))
    # output = output_q.get()  
    
    # print(output)
    # print()
    # print(output[0].shape)
