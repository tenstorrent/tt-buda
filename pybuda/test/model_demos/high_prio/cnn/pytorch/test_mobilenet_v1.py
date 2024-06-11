# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

import os
import torch.nn as nn

# SPDX-FileCopyrightText: Copyright (c) 2017 LoRnaTang
#
# SPDX-License-Identifier: Apache-2.0
# https://github.com/Lornatang/MobileNetV1-PyTorch
class Conv(nn.Module):
    """
    Conv block is convolutional layer followed by batch normalization and ReLU activation
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        use_relu6=False,
    ):
        super().__init__()
        self.layers = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
        ]

        if use_relu6:
            self.layers.append(nn.ReLU6(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.model(input)


class Conv_dw_Conv(nn.Module):
    """
    Conv_dw is depthwise (dw) convolution layer followed by batch normalization and ReLU activation.
    Conv_dw_Conv is Conv_dw block followed by Conv block.
    Implemented Conv_dw_Conv instead of Conv_dw since in MobleNet, every Conv_dw is followed by Conv
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        use_relu6=False,
    ):
        super().__init__()
        self.layers = [
            nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size,
                stride,
                padding,
                bias=False,
                groups=in_channel,
            ),
            nn.BatchNorm2d(in_channel),
        ]
        if use_relu6:
            self.layers.append(nn.ReLU6(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(
            Conv(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                use_relu6=use_relu6,
            )
        )
        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.model(input)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, use_relu6=False):
        super().__init__()

        self.num_classes = num_classes

        self.model = nn.Sequential(
            Conv(3, 32, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(32, 64, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(64, 128, kernel_size=3, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(
                128, 128, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                128, 256, kernel_size=3, stride=2, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                256, 256, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                256, 512, kernel_size=3, stride=2, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                512, 512, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                512, 512, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                512, 512, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                512, 512, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                512, 512, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                512, 1024, kernel_size=3, stride=2, use_relu6=use_relu6
            ),
            Conv_dw_Conv(
                1024, 1024, kernel_size=3, stride=1, use_relu6=use_relu6
            ),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input):
        x = self.model(input)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        out = self.fc(x)

        return out
    
    
def generate_model_mobilenetV1_base_custom_pytorch(test_device, variant):
    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBUDA module from PyTorch model
    model = MobileNetV1(9)
    tt_model = pybuda.PyTorchModule("mobilenet_v1", model)

    input_shape = (1, 3, 64, 64)
    
    image_tensor = torch.rand(*input_shape)
    
    
    return tt_model, [image_tensor], {}




def test_mobilenetv1_basic(test_device):
    model, inputs, _ = generate_model_mobilenetV1_base_custom_pytorch(
        test_device, None,
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
        )
    )

import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

def generate_model_mobilenetv1_imgcls_hf_pytorch(test_device, variant):
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
    tt_model = pybuda.PyTorchModule("mobilenet_v1__hf_075_192", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values
    
    return tt_model, [image_tensor], {}



def test_mobilenetv1_192(test_device):
    model, inputs, _ = generate_model_mobilenetv1_imgcls_hf_pytorch(
        test_device, "google/mobilenet_v1_0.75_192",
    )

    # Run inference on Tenstorrent device
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
    
def generate_model_mobilenetV1I224_imgcls_hf_pytorch(test_device, variant):
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained,
        variant
    )
    model = download_model(AutoModelForImageClassification.from_pretrained,
        variant
    )
    tt_model = pybuda.PyTorchModule("mobilenet_v1__hf_1_224", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values
    
    return tt_model, [image_tensor], {}
    
    
def test_mobilenetv1_224(test_device):
    model, inputs, _ = generate_model_mobilenetV1I224_imgcls_hf_pytorch(
        test_device, "google/mobilenet_v1_1.0_224",
    )

    # Run inference on Tenstorrent device
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
