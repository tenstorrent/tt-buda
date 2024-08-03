# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from test.utils import download_model
import timm
import pytest
import urllib
from PIL import Image
import torchvision.models as models
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger
import torchvision
from torchvision.models import efficientnet_b4, efficientnet_b0, EfficientNet_B4_Weights, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

import pybuda
from pybuda import VerifyConfig
from pybuda import CompileDepth
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.backend import verify_module

## https://huggingface.co/docs/timm/models/efficientnet

variants = [
    "efficientnet_b0",
    "efficientnet_b4",
    # "hf_hub:timm/efficientnet_b0.ra_in1k",
    # "hf_hub:timm/efficientnet_b4.ra2_in1k",
    # "hf_hub:timm/efficientnet_b5.in12k_ft_in1k",
    # "hf_hub:timm/tf_efficientnet_b0.aa_in1k",
    # "hf_hub:timm/efficientnetv2_rw_s.ra2_in1k",
    # "hf_hub:timm/tf_efficientnetv2_s.in21k",
]

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

@pytest.mark.parametrize("variant", variants)
def test_efficientnet_timm(variant, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with piepgen and blobgen errors")

    # Configuration
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

    pcc_value = 0.94
    if variant == "efficientnet_b0":
        # Solves issue for bigger conv layers in the middle of the graph
        if test_device.arch == BackendDevice.Wormhole_B0:
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_173"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_225"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_488"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_541"] = 5
            compiler_cfg.balancer_op_override("conv2d_68.dc.matmul.12", "t_stream_shape", (7,1))
            os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    elif variant == "efficientnet_b4":
        if test_device.arch == BackendDevice.Wormhole_B0:
            pcc_value = 0.92
            compiler_cfg.amp_level = 1
            compiler_cfg.default_df_override=pybuda.DataFormat.Float16_b
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Load model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()
    pybuda_model = pybuda.PyTorchModule("pt_effnet_timm", framework_model)

    # Load and pre-process image
    try:
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        img_tensor = transform(img).unsqueeze(0)
    except: 
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224)

    # Sanity run
    # cpu_output = framework_model(img_tensor)
    # cpu_probabilities = torch.nn.functional.softmax(cpu_output[0], dim=0)
    # cpu_top5_prob, cpu_top5_catid = torch.topk(cpu_probabilities, 5)
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
            pcc=pcc_value,
        ),
    )


variants = [
    models.efficientnet_b0,
    # models.efficientnet_b1,
    # models.efficientnet_b2,
    # models.efficientnet_b3,
    models.efficientnet_b4,
    # models.efficientnet_b5,
    # models.efficientnet_b6,
    # models.efficientnet_b7,
]


@pytest.mark.parametrize("variant", variants)
def test_efficientnet_torchvision(variant, test_device):
    if test_device.arch == BackendDevice.Grayskull and variant == models.efficientnet_b4:
        pytest.skip("B4 hanging on GS since f0966d4000")
        
    if test_device.arch == BackendDevice.Grayskull and variant == models.efficientnet_b0:
        pytest.skip("Error! The overlay blob for chip_0__y_7__x_1 does not fit, the max size is 73600, however we tried to allocate 345716.")
    
    # Configuration
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False # Until #844 is resolved
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if variant == models.efficientnet_b0:
        # Solves issue for bigger conv layers in the middle of the graph
        if test_device.arch == BackendDevice.Wormhole_B0:
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_67"] = 3
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_170"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_221"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_428"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_479"] = 5
            compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_531"] = 5
            os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
        
    elif variant == models.efficientnet_b4:
        # Solves issue for bigger conv layers in the middle of the graph
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_311"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_362"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_414"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_466"] = 5
        #
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_829"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_880"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_932"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_984"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1036"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1088"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1140"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1191"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1243"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1295"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1347"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1399"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1451"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1503"] = 5

        compiler_cfg.balancer_op_override("conv2d_1625.dc.matmul.8", "t_stream_shape", (1,1)) # PIPEGEN-ERROR
        compiler_cfg.balancer_op_override("conv2d_104.dc.matmul.12", "t_stream_shape", (7,1)) # blobgen error
        os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"


    # Load model
    if variant == models.efficientnet_b0:
        framework_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif variant == models.efficientnet_b4:
        framework_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    framework_model.eval()
    pybuda_model = pybuda.PyTorchModule("pt_effnet_torchvis", framework_model)

    # Load and pre-process image
    try:
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning("Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date")
        img_tensor = torch.rand(1, 3, 224, 224) 

    # Sanity run
    # cpu_output = framework_model(img_tensor)
    # cpu_probabilities = torch.nn.functional.softmax(cpu_output[0], dim=0)
    # cpu_top5_prob, cpu_top5_catid = torch.topk(cpu_probabilities, 5)
    # for i in range(cpu_top5_prob.size(0)):
    #     print(cpu_top5_catid[i].item(), cpu_top5_prob[i].item())

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
