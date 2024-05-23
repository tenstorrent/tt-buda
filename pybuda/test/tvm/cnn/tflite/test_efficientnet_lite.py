# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import torch.nn as nn
from PIL import Image

from pybuda import (
    TFLiteModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from tvm import relay
import tflite
import tensorflow as tf
import pybuda


def test_efficientnet_lite0(test_device):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True

    pybuda.config.override_op_size("conv2d_29.dc.sparse_matmul.7.dc.sparse_matmul.1.lc2", (7, 1))

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite0-fp32.tflite"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_18"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_23"] = 5


    module = TFLiteModule("tflite_efficientnet_lite0", tflite_path)

    input_shape = (1, 224, 224, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )

def test_efficientnet_lite4(test_device):
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0 or test_device.arch == pybuda.BackendDevice.Blackhole:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_conv_prestride = True
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"

    #Fusing disabled due to tenstorrent/pybuda#789
    compiler_cfg.enable_auto_fusing=False

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16}"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite4-fp32.tflite"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_93"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_99"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_105"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_111"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_117"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_128"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_134"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_140"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_146"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_152"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_158"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_164"] = 5


    module = TFLiteModule("tflite_efficientnet_lite4", tflite_path)

    input_shape = (1, 320, 320, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )
