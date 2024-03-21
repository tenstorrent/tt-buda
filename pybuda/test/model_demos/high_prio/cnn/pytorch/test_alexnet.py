# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from test.utils import download_model
import torch
import pytest
from PIL import Image
from torchvision import transforms
from loguru import logger

import pybuda
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType
from pybuda.verify.backend import verify_module
from pytorchcv.model_provider import get_model as ptcv_get_model


@pytest.mark.skip(reason="Not supported")
def test_alexnet_torchhub(test_device):
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    os.environ["PYBUDA_CONV2D_SPARSE_SECOND"] = "1"

    # Load model
    framework_model = download_model(
        torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True
    )
    framework_model.eval()
    pybuda_model = pybuda.PyTorchModule("pt_alexnet_torchhub", framework_model)

    # Load and pre-process image
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)

    # Sanity run
    # os.system(
    #     "wget -nc https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    # )
    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]
    # cpu_out = framework_model(img_tensor)
    # cpu_prob = torch.nn.functional.softmax(cpu_out[0], dim=0)
    # cpu_top5_prob, cpu_top5_catid = torch.topk(cpu_prob, 5)
    # for i in range(cpu_top5_prob.size(0)):
    #     print(categories[cpu_top5_catid[i]], cpu_top5_prob[i].item())

    # Verify
    verify_module(
        pybuda_model,
        input_shapes=[(img_tensor.shape,)],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.85,
        ),
    )


@pytest.mark.skip(reason="Not supported")
def test_alexnet_osmr(test_device):
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    os.environ["PYBUDA_CONV2D_SPARSE_SECOND"] = "1"

    # Load model
    framework_model = download_model(ptcv_get_model, "alexnet", pretrained=True)
    framework_model.eval()
    pybuda_model = pybuda.PyTorchModule("pt_alexnet_osmr", framework_model)

    # Load and pre-process image
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)

    # Sanity run
    os.system(
        "wget -nc https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    cpu_out = framework_model(img_tensor)
    cpu_prob = torch.nn.functional.softmax(cpu_out[0], dim=0)
    cpu_top5_prob, cpu_top5_catid = torch.topk(cpu_prob, 5)
    for i in range(cpu_top5_prob.size(0)):
        print(categories[cpu_top5_catid[i]], cpu_top5_prob[i].item())

    # Verify
    verify_module(
        pybuda_model,
        input_shapes=[(img_tensor.shape,)],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.85,
        ),
    )
