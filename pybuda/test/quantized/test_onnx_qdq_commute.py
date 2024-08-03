# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import onnx
import onnxruntime as ort
import pytest
import torch
import numpy as np

from pybuda import (
    BackendType,
    VerifyConfig,
)
import pybuda 
from pybuda.config import _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_int8_simple_conv(test_device):

    if test_device.arch != pybuda.BackendDevice.Wormhole_B0:
        pytest.skip('Currently works only on Wormhole_B0')

    # Load ONNX model
    load_path = "third_party/confidential_customer_models/quantized/simple_conv.onnx"
    if not os.path.exists(load_path):
        raise RuntimeError("Model not found")
    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule("int8_simple_conv", model, load_path)

    # Define inputs
    input_shape = (1, 3, 32, 32)
    input_tensor = torch.rand(input_shape)

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    #compiler_cfg.enable_auto_fusing = False
    compiler_cfg.dont_fuse(['dequantize_18.dc.buda_dequantize.3', 'dequantize_8.dc.buda_dequantize.3'])
    
    # Sanity run
    session = ort.InferenceSession(load_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: np.random.randn(*input_shape).astype(np.float32)})
    print("Sanity run output:", output)

    # Compile and verify
    pcc = 0.97 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        tt_model,
        input_shapes=([input_shape]),
        inputs=([input_tensor]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_all=True,
            verify_pipeline_result_vs_framework=True,
            verify_pybuda_codegen_vs_framework=True,
            pcc=pcc
        ),
    )


def test_int8_skip_conv(test_device):

    if test_device.arch != pybuda.BackendDevice.Wormhole_B0:
        pytest.skip('Currently works only on Wormhole_B0')

    # Load ONNX model
    load_path = "third_party/confidential_customer_models/quantized/skip_conv.onnx"
    if not os.path.exists(load_path):
        raise RuntimeError("Model not found")
    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule("int8_skip_conv", model, load_path)

    # Define inputs
    input_shape = (1, 3, 32, 32)
    input_tensor = torch.rand(input_shape)

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    #compiler_cfg.enable_auto_fusing = False
    compiler_cfg.dont_fuse(['dequantize_18.dc.buda_dequantize.3', 'dequantize_8.dc.buda_dequantize.3'])
    
    # Sanity run
    session = ort.InferenceSession(load_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: np.random.randn(*input_shape).astype(np.float32)})
    print("Sanity run output:", output)

    # Compile and verify
    pcc = 0.97 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        tt_model,
        input_shapes=([input_shape]),
        inputs=([input_tensor]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_all=True,
            verify_pipeline_result_vs_framework=True,
            verify_pybuda_codegen_vs_framework=True,
            pcc=pcc
        ),
    )

