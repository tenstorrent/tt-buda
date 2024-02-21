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

def test_int8_onnx_vit_calibrated(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Download ONNX model
    save_path = "third_party/confidential_customer_models/quantized/vit-Int8-calibrated.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_calibrated",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = DataFormat.Float32

    # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
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
            test_kind=TestKind.INFERENCE,
        ),
    )
