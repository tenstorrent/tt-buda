# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4

# STEP 0: import PyBuda library
import pytest
import pybuda
import os
import urllib
from loguru import logger
from test.utils import download_model
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
torch.multiprocessing.set_sharing_strategy("file_system") 

import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy


def generate_model_inceptionV4_imgcls_osmr_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{694:704, 676:704, 167:182, 158:160, 39:48}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "158"
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    compiler_cfg.balancer_op_override("_fused_op_4", "t_stream_shape", (158,1)) # TM error
    compiler_cfg.balancer_op_override("_fused_op_7", "t_stream_shape", (158,1)) # TM error
    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_op_override("conv2d_551.dc.sparse_matmul.10.dc.sparse_matmul.1.lc2", "grid_shape", (1,4))
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.balancer_op_override("_fused_op_2", "t_stream_shape", (676,1)) # TM error (ref pybuda#1527)

    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    pybuda_model = pybuda.PyTorchModule("pt_inception_v4_osmr", framework_model)
 
    # Load and pre-process image
    img_tensor = get_image()

    # Compile & Verify
    pcc = 0.91 if test_device.arch == BackendDevice.Grayskull else 0.97
    
    return pybuda_model, [img_tensor], {"pcc": pcc}


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
       img_tensor = torch.rand(1, 3, 299, 299)
   return model, img_tensor
 
def get_image():
    try:
        if not os.path.exists("dog.jpg"):
            torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
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
        img_tensor = torch.rand(1, 3, 299, 299)
    return img_tensor


def test_inception_v4_osmr_pytorch(test_device):
    model, inputs, other = generate_model_inceptionV4_imgcls_osmr_pytorch(
        test_device, "inceptionv4",
    )

    if test_device.arch == BackendDevice.Blackhole:
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.balancer_policy = "Ribbon"
        compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
        compiler_cfg.enable_auto_fusing = False
        compiler_cfg.place_on_new_epoch("multiply_35")

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=other["pcc"],
            # padding overrides cause tensor size mismatch during verification: tenstorrent/pybuda#627
            verify_post_autograd_passes=False,
            verify_post_placer=False,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        ),
    )

def generate_model_inceptionV4_imgcls_timm_pytorch(test_device, variant):
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{694:704, 676:704, 167:182, 158:160, 39:48}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "158"
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    compiler_cfg.balancer_op_override("_fused_op_4", "t_stream_shape", (158,1)) # TM error
    compiler_cfg.balancer_op_override("_fused_op_7", "t_stream_shape", (158,1)) # TM error
    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_op_override("conv2d_551.dc.sparse_matmul.10.dc.sparse_matmul.1.lc2", "grid_shape", (1,4))
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.balancer_op_override("_fused_op_2", "t_stream_shape", (676,1)) # TM error (ref pybuda#1527)

    # Load model & Preprocess image
    framework_model, img_tensor = download_model(preprocess_timm_model, variant)
    pybuda_model = pybuda.PyTorchModule("pt_inception_v4_timm", framework_model)

    # Compile & Verify
    pcc = 0.96 if test_device.arch == BackendDevice.Grayskull else 0.97
    
    return pybuda_model, [img_tensor], {}


def test_inception_v4_timm_pytorch(test_device):
    model, inputs, _ = generate_model_inceptionV4_imgcls_timm_pytorch(
        test_device, 'inception_v4',
    )

    if test_device.arch == BackendDevice.Blackhole:
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.balancer_policy = "Ribbon"
        compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
        compiler_cfg.enable_auto_fusing = False
        compiler_cfg.place_on_new_epoch("multiply_35")

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.97,
            # padding overrides cause tensor size mismatch during verification: tenstorrent/pybuda#627
            verify_post_autograd_passes=False,
            verify_post_placer=False,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        ),
    )
