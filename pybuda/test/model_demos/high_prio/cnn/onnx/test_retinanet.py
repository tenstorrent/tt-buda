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
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

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
