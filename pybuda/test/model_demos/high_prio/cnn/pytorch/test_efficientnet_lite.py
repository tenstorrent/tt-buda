# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## EfficientNet V1 demo
import pytest

# STEP 0: import PyBuda library
import pybuda
import os

import math
import torch
from torch import nn
import torch.functional as F

import urllib
from PIL import Image
from torchvision import transforms
import json

from pybuda.op.eval import compare_tensor_to_golden
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

## https://github.com/RangiLyu/EfficientNet-Lite/
from test.model_demos.utils.cnn.pytorch.saved.efficientnet_lite import src_efficientnet_lite as efflite

#############
def get_image_tensor(wh):
    # Image processing
    tfms = transforms.Compose([
        transforms.Resize(wh),     
        transforms.CenterCrop(wh), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img_tensor = tfms(Image.open('pybuda/test/model_demos/utils/cnn/pytorch/images/img.jpeg')).unsqueeze(0)
    return img_tensor
######

def test_efficientnet_lite_0_pytorch(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Compiles but gives: \"Forward error: Failed while running fwd program\" when running")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    elif test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # STEP 2: Model load in PyBuda 
    model_name = 'efficientnet_lite0'
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("third_party/confidential_customer_models/model_2/pytorch/efficientnet_lite/weights/efficientnet_lite0.pth")
    model.eval() 
     
    tt_model = pybuda.PyTorchModule("pt_effnet_lite0", model)

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)
    
    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.94,
        )
    )

def test_efficientnet_lite_1_pytorch(test_device): 

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Backend compile failed")

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.amp_level = 2
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    # STEP 2: Model load in PyBuda 
    model_name = 'efficientnet_lite1'
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("third_party/confidential_customer_models/model_2/pytorch/efficientnet_lite/weights/efficientnet_lite1.pth")
    model.eval() 
     
    tt_model = pybuda.PyTorchModule("pt_effnet_lite1", model)

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)

    pcc = 0.9 if test_device.arch == BackendDevice.Grayskull else 0.94
    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_post_placer=False,
            pcc=pcc,
        )
    )
 
def test_efficientnet_lite_2_pytorch(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Backend compile failed")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.amp_level = 2
    compiler_cfg.balancer_op_override("conv2d_99.dc.conv2d.1.dc.matmul.12", "grid_shape", (7,5))
    compiler_cfg.balancer_op_override("conv2d_142.dc.conv2d.1.dc.matmul.12", "grid_shape", (7,5))
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{529:544}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "529"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"

    # STEP 2: Model load in PyBuda 
    model_name = 'efficientnet_lite2'
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("third_party/confidential_customer_models/model_2/pytorch/efficientnet_lite/weights/efficientnet_lite2.pth")
    model.eval() 
 
    tt_model = pybuda.PyTorchModule("pt_effnet_lite2", model)

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)
 
    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        )
    )

def test_efficientnet_lite_3_pytorch(test_device):

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Fails with: Error! fork_stream_ids exceeds max fork allowed for chip_0__y_3__x_2, stream_id=24")

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{613:640, 39:48, 11:12}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "613"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_MM"] = "{26:27}"
    elif test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "FastCut"
 
    # STEP 2: Model load in PyBuda 
    model_name = 'efficientnet_lite3'
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("third_party/confidential_customer_models/model_2/pytorch/efficientnet_lite/weights/efficientnet_lite3.pth")
    model.eval() 
     
    tt_model = pybuda.PyTorchModule("pt_effnet_lite3", model)

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)

    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )

def test_efficientnet_lite_4_pytorch(test_device):

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Fails with: Error! The overlay blob for chip_0__y_7__x_1 does not fit, the max size is 130944, however we tried to allocate 133168.")

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{46:48}"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
        compiler_cfg.amp_level = 1
        os.environ["PYBUDA_RIBBON2"] = "1"
    elif test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_CONCAT"] = "{51:54, 11:16, 6:8, 5:8}"
 
    # STEP 2: Model load in PyBuda 
    model_name = 'efficientnet_lite4'
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("third_party/confidential_customer_models/model_2/pytorch/efficientnet_lite/weights/efficientnet_lite4.pth")
    model.eval() 
 
    tt_model = pybuda.PyTorchModule("pt_effnet_lite4", model)

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)

    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.92,
        )
    )

