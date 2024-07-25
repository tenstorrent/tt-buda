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
from pybuda._C import MathFidelity
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import _get_global_compiler_config

def test_onnx_qdq_regnet_y(test_device):
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull does not support quantized models")

    save_path = "third_party/confidential_customer_models/bos/bos_onnx_062524/priorityA/regnet_y_32gf/regnet_y_32gf_ptq_qdq.onnx"

    onnx_model = onnx.load(save_path)
    # onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_qdq_regnet_y",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.retain_tvm_python_files = True
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
            # verify_all=True,
        ),
        input_params=[{"requires_grad": False}]
    )