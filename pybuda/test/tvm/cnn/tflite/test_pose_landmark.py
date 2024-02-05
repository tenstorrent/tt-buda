# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

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
from pybuda.verify.config import TestKind
from tvm import relay
import tflite
import tensorflow as tf




def test_pose_landmark_lite(test_device):
    pytest.skip("Resize2d DenseMM const too big")
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_lite.tflite"


    module = TFLiteModule("tflite_pose_landmark_light", tflite_path)

    input_shape = (1, 256, 256, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_pose_landmark_heavy(test_device):
    pytest.skip("Resize2d DenseMM const too big")
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_heavy.tflite"


    module = TFLiteModule("tflite_pose_landmark_heavy", tflite_path)

    input_shape = (1, 256, 256, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )
