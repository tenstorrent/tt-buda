# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os

from PIL import Image
import numpy as np

import onnx
import torch
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind


########
# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data
######### 

def test_yolov3_tiny_onnx(test_device):
        
    pytest.skip("While loop in model, not supported yet")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"

    # STEP 2: Create PyBuda module from PyTorch model 
    load_path = "third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/tiny-yolov3-11.onnx"
    model = onnx.load(load_path)  
    tt_model = pybuda.OnnxModule("onnx_yolov3_tiny", model, load_path) 

    # Image preprocessing
    pil_img = Image.open("third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/carvana.jpg")
    # input
    image_data = preprocess(pil_img)
    image_size = np.array([pil_img.size[1], pil_img.size[0]], dtype=np.int32).reshape(1, 2)
    image_data = torch.from_numpy(image_data).type(torch.float)
    image_size = torch.from_numpy(image_size).type(torch.float)

    verify_module(
        tt_model,
        input_shapes=[image_data.shape, image_size.shape],
        inputs=[(image_data, image_size)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
 
def test_yolov3_onnx(test_device):
    pytest.skip("While loop in model, not supported yet")
    
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"

    # STEP 2: Create PyBuda module from PyTorch model 
    load_path = "third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/yolov3-10.onnx"
    model = onnx.load(load_path)  
    tt_model = pybuda.OnnxModule("onnx_yolov3_tiny", model, load_path) 

    # Image preprocessing
    pil_img = Image.open("third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/carvana.jpg")
    # input
    image_data = preprocess(pil_img)
    image_size = np.array([pil_img.size[1], pil_img.size[0]], dtype=np.int32).reshape(1, 2)
    image_data = torch.from_numpy(image_data).type(torch.float)
    image_size = torch.from_numpy(image_size).type(torch.float) 
     
    verify_module(
        tt_model,
        input_shapes=[image_data.shape, image_size.shape],
        inputs=[(image_data, image_size)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )

