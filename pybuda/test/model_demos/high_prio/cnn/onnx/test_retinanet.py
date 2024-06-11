# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# STEP 0: import PyBuda library
import pytest

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig, PyTorchModule
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind
import pybuda
import os

import onnx
import torch
import tensorflow as tf

from PIL import Image
import numpy as np

import requests
from torchvision import transforms

## https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet

########
def img_preprocess(scal_val=1):
    pil_img = Image.open("pybuda/test/model_demos/utils/cnn/onnx/images/carvana.jpg")
    scale=scal_val
    w, h = pil_img.size
    print("----", w, h)
    newW, newH = int(scale * w), int(scale * h)
    newW, newH = 640, 480 
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(pil_img, dtype=np.float32)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))
    if (img > 1).any():
        img = img / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img

######### 
 
def test_retinanet_r101_640x480_onnx(test_device):
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{76*1024}"
    os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    # Temp mitigations for net2pipe errors, should be removed.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_356"] = 3

    # STEP 2: Create PyBuda module from PyTorch model 
    load_path = "third_party/confidential_customer_models/model_2/onnx/retinanet/retinanet-9.onnx"
    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule("onnx_retinanet", model, load_path) 

    # Image preprocessing
    img_tensor = img_preprocess()

    # STEP 3: Run inference on Tenstorrent device
    pcc = 0.97 if test_device.arch == BackendDevice.Grayskull and test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        tt_model, 
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            pcc=pcc,
        )
    )

def img_preprocessing():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img

variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn152fpn",
]

@pytest.mark.parametrize("variant", variants)
def test_retinanet_onnx(variant, test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "73728"

        if variant == "retinanet_rn18fpn":
            compiler_cfg.place_on_new_epoch("conv2d_117.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_82.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_60.dc.matmul.11", "grid_shape", (1,1))

        elif variant == "retinanet_rn34fpn":
            compiler_cfg.place_on_new_epoch("conv2d_157.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_122.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_100.dc.matmul.11", "grid_shape", (1,1))
        
        elif variant == "retinanet_rn50fpn":
            compiler_cfg.place_on_new_epoch("conv2d_190.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_155.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_133.dc.matmul.11", "grid_shape", (1,1))

        elif variant == "retinanet_rn152fpn":
            compiler_cfg.place_on_new_epoch("conv2d_428.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_393.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_371.dc.matmul.11", "grid_shape", (1,1))
    
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "69632"
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

        if variant == "retinanet_rn18fpn":
            compiler_cfg.balancer_op_override("conv2d_82.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_60.dc.matmul.11", "t_stream_shape", (1,1))

        elif variant == "retinanet_rn34fpn":
            compiler_cfg.balancer_op_override("conv2d_122.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_100.dc.matmul.11", "t_stream_shape", (1,1))

        elif variant == "retinanet_rn50fpn":
            compiler_cfg.balancer_op_override("conv2d_155.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_133.dc.matmul.11", "t_stream_shape", (1,1))

        elif variant == "retinanet_rn152fpn":
            compiler_cfg.balancer_op_override("conv2d_393.dc.matmul.11", "t_stream_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_371.dc.matmul.11", "t_stream_shape", (1,1))
 
    # Prepare model
    load_path = (
        f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    )
    model_name = f"onnx_{variant}"
    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule(model_name, model, load_path)

    # Prepare input
    input_batch = img_preprocessing()

    # Inference
    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
