# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import urllib
from test.utils import download_model
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pybuda
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.backend import verify_module

variants = [
    "ghostnet_050",
    "ghostnet_100",
    "ghostnet_130",
]


@pytest.mark.parametrize("variant", variants)
def test_ghostnet_timm(variant, test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Failing with: HLKC unpack compile failed.")
    
    if variant == "ghostnet_130":
        pytest.skip("Skip ghostnet_130 due to hang on device")
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()
    pybuda_model = pybuda.PyTorchModule("pt_ghostnet_timm", framework_model)

    # Load and pre-process image
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    img_tensor = transform(img).unsqueeze(0)

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
