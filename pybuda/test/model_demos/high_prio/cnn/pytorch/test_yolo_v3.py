# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys

from PIL import Image

import torch
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

# https://github.com/holli/yolov3_pytorch
sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))

from yolo_v3.holli_src import utils
from yolo_v3.holli_src.yolo_layer import *
from yolo_v3.holli_src.yolov3_tiny import *
from yolo_v3.holli_src.yolov3 import *


def generate_model_yolotinyV3_imgcls_holli_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
    model.load_state_dict(torch.load('third_party/confidential_customer_models/model_2/pytorch/yolo_v3/weights/yolov3_tiny_coco_01.h5'))
    model.eval()

    # STEP 2: Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pytorch_yolov3_tiny_holli", model)

    sz = 512
    imgfile = "third_party/confidential_customer_models/model_2/pytorch/yolo_v3/person.jpg"
    img_org = Image.open(imgfile).convert('RGB')
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    return tt_model, [img_tensor], {}


def test_yolov3_tiny_holli_pytorch(test_device):
    model, inputs, _ = generate_model_yolotinyV3_imgcls_holli_pytorch(
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
            pcc=0.97,
        )
    )


def generate_model_yoloV3_imgcls_holli_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    model = Yolov3(num_classes=80)
    model.load_state_dict(torch.load('third_party/confidential_customer_models/model_2/pytorch/yolo_v3/weights/yolov3_coco_01.h5', map_location=torch.device('cpu')))
    model.eval()

    # STEP 2: Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pytorch_yolov3_holli", model)

    sz = 512
    imgfile = "third_party/confidential_customer_models/model_2/pytorch/yolo_v3/person.jpg"
    img_org = Image.open(imgfile).convert('RGB')
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    pcc = 0.9
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        pcc = 0.86

    return tt_model, [img_tensor], {"pcc": pcc}


def test_yolov3_holli_pytorch(test_device):
    model, inputs, other = generate_model_yoloV3_imgcls_holli_pytorch(
        test_device, None,
    )

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"

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
            pcc=other["pcc"],
        )
    )

def test_yolov3_holli_pytorch_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"
    model, inputs, other = generate_model_yoloV3_imgcls_holli_pytorch(
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
            pcc=other["pcc"],
        )
    )

