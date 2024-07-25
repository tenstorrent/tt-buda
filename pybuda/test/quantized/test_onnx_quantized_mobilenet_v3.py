# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import onnx
import pybuda.module
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


def find_init(init_names, model):
    initializers = []
    for init in model.graph.initializer:
        if init.name in init_names:
            initializers.append(init)
    return initializers

def find_node(node_name, model):
    for node in model.graph.node:
        if node.name == node_name:
            return node
    return None

def find_idx(node_name, model):
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            return i
    return None

def test_onnx_qdq_mobilenet_v3(test_device):
    # pytest.skip("Models not yet uploaded")
    pytest.skip("WIP")
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull does not support quantized models")

    save_path = "third_party/confidential_customer_models/bos/bos_onnx_062524/priorityA/mobilenetv3/mobilenetv3_ptq_qdq.onnx"

    onnx_model = onnx.load(save_path)

    split_node_name = '/features/features.1/add2/Add'
    split_node_idx = find_idx(split_node_name, onnx_model)
    nodes_to_remove = []
    for i in range(split_node_idx + 1, len(onnx_model.graph.node)):
        nodes_to_remove.append(onnx_model.graph.node[i])
    for node in nodes_to_remove:
        onnx_model.graph.node.remove(node)

    # import pdb; pdb.set_trace()
    output_name = onnx_model.graph.output[0].name
    onnx_model.graph.output.pop()
    new_output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1,120,28,28])
    onnx_model.graph.output.append(new_output)
    last_node = find_node(split_node_name, onnx_model)
    last_node.output[0] = output_name


    

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "./chopped_mobilenetv3.onnx")
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_mobilenet_v3",
        onnx_model,
        "./chopped_mobilenetv3.onnx",
    )

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.default_math_fidelity = MathFidelity.HiFi4
    # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    # os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    # os.environ["PYBUDA_DISABLE_DEPTHWISE_CONV2D_DECOMP"] = "1"

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
            verify_pybuda_codegen_vs_framework=True,
            # verify_tvm_compile=True,
            # verify_all=True,
        ),
    )