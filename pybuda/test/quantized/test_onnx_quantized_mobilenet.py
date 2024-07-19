# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import onnx
import pytest
import numpy as np
import onnxruntime
import torch
import pybuda
from pybuda import (
    OnnxModule,
    VerifyConfig,
    DataFormat,
    BackendDevice,
    BackendType,
)
from pybuda._C import MathFidelity
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import _get_global_compiler_config

def test_onnx_quantized_mb_v2_depth(test_device):
    # Skip test on blackhole until we have support for quantized models on blackhole pybuda#2700
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    # Download ONNX model
    save_path = "third_party/confidential_customer_models/quantized/mb_v2_depthwise-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mb_v2_depthwise",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    if test_device.devtype == BackendType.Silicon:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{80*1024}"

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    # Compile and verify
    verify_module(
        pybuda_onnx_model,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled = False if test_device.devtype == BackendType.Silicon else True,
            # verify_pybuda_codegen_vs_framework=True,
            # verify_all=True
        ),
    )


def test_onnx_quantized_mb_v2(test_device):
    # Skip test on blackhole until we have support for quantized models on blackhole pybuda#2700
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    # Download ONNX model
    save_path = "third_party/confidential_customer_models/quantized/mobilenet_v2-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mb_v2",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    compiler_cfg.place_on_new_epoch("conv2d_122.dc.reshape.0.dc.sparse_matmul.14.lc2")
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{80*1024}"
    if test_device.devtype == BackendType.Silicon:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{96*1024}"

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    # Compile and verify
    verify_module(
        pybuda_onnx_model,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled = False if test_device.devtype == BackendType.Silicon else True,
            # verify_pybuda_codegen_vs_framework=True,
            # verify_all=True
        ),
    )

def test_onnx_qdq_mobilenet(test_device):
    # pytest.skip("Models not yet uploaded")
    pytest.skip("WIP")
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull does not support quantized models")

    save_path = "third_party/confidential_customer_models/bos/mobilenetv2_ptq_qdq.onnx"

    onnx_model = onnx.load(save_path)
    # onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mobilenet_v2",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_math_fidelity = MathFidelity.HiFi4
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    # Compile and verify
    verify_module(
        pybuda_onnx_model,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            # verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
        ),
    )