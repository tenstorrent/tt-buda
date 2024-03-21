# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import os
import torch
import torch.nn as nn
from PIL import Image

from pybuda import (
    TFLiteModule,
    VerifyConfig,
    BackendType,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind, BackendDevice
from tvm import relay
import tflite
import tensorflow as tf
import pybuda
import os


def test_pose_landmark_lite_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_SPLIT_RESIZE2D"] = "128"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_MAX_CONCAT_INPUTS"] = "6"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_single_buffer_fallback = True

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_lite.tflite"

    module = TFLiteModule("tflite_pose_landmark_light", tflite_path)

    input_shape = (1, 256, 256, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            atol=0.001,
            pcc=0.94,
            enabled=False,
        ),
    )


def test_pose_landmark_heavy_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_SPLIT_RESIZE2D"] = "128"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_MAX_CONCAT_INPUTS"] = "6"
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.amp_level = 1
    compiler_cfg.enable_single_buffer_fallback = True

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_heavy.tflite"


    module = TFLiteModule("tflite_pose_landmark_heavy", tflite_path)

    input_shape = (1, 256, 256, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            atol=0.001,
            pcc=0.9,
            enabled=False,
        ),
    )

@pytest.mark.skip(reason="Not supported")
def test_pose_landmark_lite(test_device):
    if test_device.devtype == BackendType.Silicon:
        pytest.skip("silicon run hangs")
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_55"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_61"] = 5
        compiler_cfg.place_on_new_epoch("conv2d_61.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")  # dram-queue
        compiler_cfg.balancer_op_override("conv2d_21.dc.matmul.11", "grid_shape", (4,3))  # blobgen
        compiler_cfg.balancer_op_override("conv2d_26.dc.matmul.11", "grid_shape", (4,5))
    elif test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_CONCAT"] = "{11:12}"
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_21"] = 5
        compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_26"] = 5

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_lite.tflite"


    module = TFLiteModule("tflite_pose_landmark_light", tflite_path)

    input_shape = (1, 256, 256, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )

@pytest.mark.skip(reason="Not supported")
def test_pose_landmark_heavy(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"

    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_38"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_44"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_50"] = 5
    compiler_cfg.amp_level = 1

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_heavy.tflite"
    module = TFLiteModule("tflite_pose_landmark_heavy", tflite_path)

    input_shape = (1, 256, 256, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
