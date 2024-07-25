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


def test_onnx_qdq_mobilenet_v2(test_device):
    # pytest.skip("Models not yet uploaded")
    # pytest.skip("WIP")
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull does not support quantized models")

    save_path = "third_party/confidential_customer_models/bos/bos_onnx_062524/priorityA/mobilenetv2/mobilenetv2_ptq_qdq.onnx"

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
    os.environ["PYBUDA_DISABLE_DEPTHWISE_CONV2D_DECOMP"] = "1"

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
            pcc=0.96
        ),
    )

class MobilenetV2QDQDepthwise(pybuda.module.PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("conv_kernel", pybuda.Parameter(*(32, 1, 3, 3), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("conv_bias", pybuda.Parameter(*(32,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_constant("const_00", shape=(1,))
        self.set_constant("const_00", torch.tensor([1.0]))
        self.add_constant("/features/features.1/conv/conv.0/scale", shape=(32,))
        self.set_constant("/features/features.1/conv/conv.0/scale", torch.tensor([0.0409439280629158]))

    def forward(self, img):

        one = self.get_constant("const_00")
        kernel = self.get_parameter("conv_kernel")
        bias = self.get_parameter("conv_bias")
        scale = self.get_constant("/features/features.1/conv/conv.0/scale")

        # Muse use int8 inputs
        img = pybuda.op.Quantize("", img, one, out_dtype=torch.int8, axis=0, zero_point=0.0)
        kernel = pybuda.op.Quantize("", kernel, scale, out_dtype=torch.int8, axis=0, zero_point=0.0)
        bias = pybuda.op.Quantize("", bias, scale, out_dtype=torch.int32, axis=0, zero_point=0.0)

        # This Conv2d will decompose. One op included in the decomposition will be "depthwise"
        out = pybuda.op.Conv2d("", img, kernel, bias, stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=32, channel_last=0)

        # Output must be float32 or the output cannot be untilized
        out = pybuda.op.Dequantize("", out, scale, out_dtype=torch.float32, axis=0, zero_point=0.0)
        
        # These are the inverse of what the output of the conv2d decompositon inserts.
        # Puth this here so that there are no tms between the depthwise and the output
        out = pybuda.op.Reshape("", out, (1, 1, 32, 12544))
        out = pybuda.op.Transpose("", out, -1, -2)
        return out
    
def test_depthwise(test_device):

    # pytest.skip("Models not yet uploaded")
    pytest.skip("Use this test to debug the int8 depthwise issue")
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Grayskull does not support quantized models")

    save_path = "third_party/confidential_customer_models/bos/bos_onnx_062524/priorityA/mobilenetv2/mobilenetv2_ptq_qdq.onnx"

    onnx_model = onnx.load(save_path)
    # onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = MobilenetV2QDQDepthwise("mbv2_depthwise")
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.retain_tvm_python_files = True
    # compiler_cfg.default_math_fidelity = MathFidelity.HiFi4
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    # compiler_cfg.place_on_new_epoch("conv2d_122.dc.reshape.0.dc.sparse_matmul.14.lc2")
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
        [(1, 32, 112, 112)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            # verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            pcc=0.96
        ),
    )