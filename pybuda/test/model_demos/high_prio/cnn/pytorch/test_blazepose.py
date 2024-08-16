# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BlazePose Demo Script - PyTorch
from pybuda.verify.config import TestKind, BackendDevice, BackendType

import os
from pybuda.verify.backend import verify_module
import pytest
import cv2
import pybuda
import torch
import sys

sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
from mediapipepytorch.blazebase import denormalize_detections, resize_pad
from mediapipepytorch.blazepose import BlazePose
from mediapipepytorch.blazepalm import BlazePalm
from mediapipepytorch.blazehand_landmark import BlazeHandLandmark
from mediapipepytorch.blazepose_landmark import BlazePoseLandmark
from mediapipepytorch.visualization import POSE_CONNECTIONS, draw_landmarks


@pytest.mark.skip(reason="Only test 1x1 grid")
def test_blazepose_detector_pytorch(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with TM ERROR (producer = conv2d_163.dc.add.11_fused_tm_op_0.dc.matmul.7, consumer = conv2d_163.dc.add.11_fused_tm_op_0.dc.matmul.12): TM order does't satisfy constraints for stacking with phased pipes, buf_size_mb must be a multiple of the total stack factor or producer t")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    # Load BlazePose Detector
    pose_detector = BlazePose()
    pose_detector.load_weights("third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/blazepose.pth")
    pose_detector.load_anchors("third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/anchors_pose.npy")

    # Load data sample
    orig_image = cv2.imread("pybuda/test/model_demos/utils/cnn/pytorch/images/girl.png")

    # Preprocess for BlazePose Detector
    _, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0

    verify_module(
        pybuda.PyTorchModule("pt_blazepose_detector", pose_detector),
        input_shapes=[(img2.shape,)],
        inputs=[(img2,)],
        verify_cfg=pybuda.VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            # pcc=0.9
        )
    )

@pytest.mark.skip(reason="Only test 1x1 grid")
def test_blazepose_regressor_pytorch(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with data mismatch")
    
    # Set PyBDUA environment variable
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"


    # Load BlazePose Landmark Regressor
    pose_regressor = BlazePoseLandmark()
    pose_regressor.load_weights("third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/blazepose_landmark.pth")

    verify_module(
        pybuda.PyTorchModule("pt_blazepose_regressor", pose_regressor),
        input_shapes=[(1, 3, 256, 256,)],
        verify_cfg=pybuda.VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            atol=0.02,
            # pcc=0.9
        )
    )

@pytest.mark.skip(reason="Not supported")
def test_blazepose_detector_pytorch_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with TM ERROR (producer = conv2d_163.dc.add.11_fused_tm_op_0.dc.matmul.7, consumer = conv2d_163.dc.add.11_fused_tm_op_0.dc.matmul.12): TM order does't satisfy constraints for stacking with phased pipes, buf_size_mb must be a multiple of the total stack factor or producer t")

    # Set PyBDUA environment variable
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    # Load BlazePose Detector
    pose_detector = BlazePose()
    pose_detector.load_weights("third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/blazepose.pth")
    pose_detector.load_anchors("third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/anchors_pose.npy")

    # Load data sample
    orig_image = cv2.imread("pybuda/test/model_demos/utils/cnn/pytorch/images/girl.png")

    # Preprocess for BlazePose Detector
    _, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0

    verify_module(
        pybuda.PyTorchModule("pt_blazepose_detector", pose_detector),
        input_shapes=[(img2.shape,)],
        inputs=[(img2,)],
        verify_cfg=pybuda.VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            # pcc=0.9
        )
    )

@pytest.mark.skip(reason="Not supported")
def test_blazepose_regressor_pytorch_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull test failing with data mismatch")
    
    # Set PyBDUA environment variable
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    # Load BlazePose Landmark Regressor
    pose_regressor = BlazePoseLandmark()
    pose_regressor.load_weights("third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/blazepose_landmark.pth")

    verify_module(
        pybuda.PyTorchModule("pt_blazepose_regressor", pose_regressor),
        input_shapes=[(1, 3, 256, 256,)],
        verify_cfg=pybuda.VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            atol=0.02,
            # pcc=0.9
        )
    )

def test_blaze_palm_pytorch_1x1(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    elif test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    elif test_device.arch == BackendDevice.Blackhole:
        os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "blackhole_1x1.yaml"

    # Set PyBDUA environment variable
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
    
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.cpu_fallback_ops = set(["concatenate"])

    # Load BlazePalm Detector
    palm_detector = BlazePalm()
    palm_detector.load_weights(
        "third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/blazepalm.pth"
    )
    palm_detector.load_anchors(
        "third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/anchors_palm.npy"
    )
    palm_detector.min_score_thresh = 0.75

    # Load data sample
    orig_image = cv2.imread("pybuda/test/model_demos/utils/cnn/pytorch/images/girl.png")

    # Preprocess for BlazePose Detector
    img1, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0

    verify_module(
        pybuda.PyTorchModule("pt_palm_detector", palm_detector),
        input_shapes=[(img2.shape,)],
        inputs=[(img2,)],
        verify_cfg=pybuda.VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            enabled=False,
        )
    )

def test_blaze_hand_pytorch_1x1(test_device):
    
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Set PyBDUA environment variable
    elif test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    elif test_device.arch == BackendDevice.Blackhole:
        os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "blackhole_1x1.yaml"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_33"] = -1
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_112"] = -1

    # Load BlazePalm Detector
    hand_regressor = BlazeHandLandmark()
    hand_regressor.load_weights(
        "third_party/confidential_customer_models/model_2/pytorch/mediapipepytorch/blazehand_landmark.pth"
    )

    verify_module(
        pybuda.PyTorchModule("pt_hand_regressor", hand_regressor),
        input_shapes=[(1, 3, 256, 256,)],
        verify_cfg=pybuda.VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            atol=0.01,
            enabled=False,
        )
    )
