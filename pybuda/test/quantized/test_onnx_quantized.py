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
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import _get_global_compiler_config

def test_onnx_quantized_mlp_gelu(test_kind, test_device):
    pytest.skip()
    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/mlp_gelu-QOperator.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mlp_gelu",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
        ),
    )

def test_onnx_quantized_mlp(test_kind, test_device):
    pytest.skip()
    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/mlp-QOperator.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mlp",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
        ),
    )


def test_onnx_quantized_conv(test_kind, test_device):
    pytest.skip()
    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/conv2d_with_bias-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_conv",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    tti_path = "int8_conv_bias.tti"
    if not os.path.exists(tti_path):
        tt_module = pybuda_onnx_model
        device = pybuda.TTDevice(
            "tt0", module=tt_module,arch=pybuda.BackendDevice.Wormhole_B0, devtype=pybuda.BackendType.Silicon)
        tti_img = device.compile_to_image(
            img_path=tti_path,
            training=False,
            sample_inputs=[torch.randn(shape) for shape in input_shape],
        )

    device_img: pybuda.TTDeviceImage = pybuda.TTDeviceImage.load_from_disk(tti_path)
    ttdevice = pybuda.TTDevice.load_image(img=device_img)

    inputs = [torch.randn(shape) for shape in input_shape]
    ttdevice.push_to_inputs(*inputs)
    output_q = pybuda.run_inference(_sequential=True)
    output = output_q.get()[0].value().detach()

    golden_output = pybuda_onnx_model.forward(*inputs)
    assert np.allclose(output, golden_output[0], atol=1e-3, rtol=1e-3)
    # # Compile and verify
    # verify_module(
    #     pybuda_onnx_model,
    #     input_shape,
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         test_kind=test_kind,
    #         verify_pybuda_codegen_vs_framework=True,
    #     ),
    # )

def test_onnx_quantized_mm_int8_no_bias(test_kind, test_device):
    pytest.skip()
    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/matmul_no_bias-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mm_int8_no_bias",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            # verify_all=True, # need to update matmul eval in buda 
        ),
    )

def test_onnx_quantized_mm_int8_bias(test_kind, test_device):
    pytest.skip()
    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/matmul_with_bias-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mm_int8_bias",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            # verify_all=True,
        ),
    )

def test_onnx_quantized_mm_uint8_no_bias(test_kind, test_device):
    pytest.skip()
    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/matmul_no_bias-UInt8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mm_uint8_no_bias",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False

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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
        ),
    )



