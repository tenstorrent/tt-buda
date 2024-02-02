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




def test_mobilenet_ssd_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override=pybuda.DataFormat.Float16_b
    compiler_cfg.cpu_fallback_ops = set(["concatenate"])

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/ssd_mobilenet_v2.tflite"

    module = TFLiteModule("tflite_mobilenet_ssd", tflite_path)

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
