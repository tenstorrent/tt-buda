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
    DataFormat,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind, BackendDevice
from tvm import relay
import tflite
import tensorflow as tf
import pybuda
import os


def test_efficientnet_lite0_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite0-fp32.tflite"

    module = TFLiteModule("tflite_efficientnet_lite0", tflite_path)

    input_shape = (1, 224, 224, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc = 0.97 if test_device.arch == BackendDevice.Grayskull else 0.99 # GS PCC = 0.975
        ),
    )

def test_efficientnet_lite4_1x1(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite4-fp32.tflite"

    module = TFLiteModule("tflite_efficientnet_lite4", tflite_path)

    input_shape = (1, 320, 320, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pipeline_result_vs_framework=False,
            verify_pybuda_codegen_vs_framework=False,
        ),
    )




def test_efficientnet_lite0(test_device):
    pytest.skip("Only test 1x1 grid")
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite0-fp32.tflite"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_18"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_23"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_51"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_57"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_69"] = 5

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{7:8, 25:32, 98:128}"
    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_CONCAT"] = "{5:8, 15:16, 3:4,21:24}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "98"
    module = TFLiteModule("tflite_efficientnet_lite0", tflite_path)

    input_shape = (1, 224, 224, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc = 0.97 if test_device.arch == BackendDevice.Grayskull else 0.99 # GS PCC = 0.975
        ),
    )

def test_efficientnet_lite1(test_device):
    pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.default_df_override = DataFormat.Float16_b

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite1-fp32.tflite"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_29"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_24"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_35"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_69"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_75"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_81"] = 5

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{113:128}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "113"

    module = TFLiteModule("tflite_efficientnet_lite1", tflite_path)

    input_shape = (1, 240, 240, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_post_placer=False,
        ),
    )

def test_efficientnet_lite2(test_device):
    pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.amp_level = 2
    compiler_cfg.default_df_override = DataFormat.Float16_b
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    # compiler_cfg.enable_conv_prestride = True

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{529:532, 35:48}"
    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_CONCAT"] = "{5:8, 17:20, 23:24, 39:40}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "133"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite2-fp32.tflite"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_0"] = 3

    module = TFLiteModule("tflite_efficientnet_lite2", tflite_path)

    input_shape = (1, 260, 260, 3)
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

def test_efficientnet_lite3(test_device):
    pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.amp_level = 2
    compiler_cfg.default_df_override = DataFormat.Float16_b
    compiler_cfg.graph_solver_self_cut_type = "FastCut"

    import os
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{613:640}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "613"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite3-fp32.tflite"


    module = TFLiteModule("tflite_efficientnet_lite3", tflite_path)

    input_shape = (1, 280, 280, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_post_placer=False,
        ),
    )

def test_efficientnet_lite4(test_device):
    pytest.skip()
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull failing with: Error! The overlay blob for chip_0__y_1__x_12 does not fit, the max size is 65408, however we tried to allocate 71240.")
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_conv_prestride = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16}"
    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_CONCAT"] = "{51:54, 11:16, 6:8, 5:8, 21:24, 30:32}"

    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite4-fp32.tflite"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_88"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_93"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_99"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_105"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_111"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_117"] = 5
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_123"] = 5
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
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pipeline_result_vs_framework=False,
            verify_pybuda_codegen_vs_framework=False,
        ),
    )
