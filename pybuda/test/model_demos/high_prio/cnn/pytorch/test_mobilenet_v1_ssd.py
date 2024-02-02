# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

import os
import pybuda
import sys

sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
from mobilenetv1_ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


def test_mobilenet_v1_ssd_pytorch_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()
    
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = (
        pybuda.config._get_global_compiler_config()
    )  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_102"] = -1
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_131"] = -1

    # Load PASCAL VOC dataset class labels
    label_path = "third_party/confidential_customer_models/model_2/pytorch/mobilenetv1_ssd/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    number_of_classes = len(class_names)

    # STEP 2: Create PyBuda module from PyTorch model
    model_path = "third_party/confidential_customer_models/model_2/pytorch/mobilenetv1_ssd/models/mobilenet-v1-ssd-mp-0_675.pth"
    net = create_mobilenetv1_ssd(number_of_classes)
    net.load(model_path)
    net.eval()

    tt_model = pybuda.PyTorchModule("pt_mobilenet_v1_ssd", net)

    input_shape = (1, 3, 300, 300)

    # STEP 3: Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
