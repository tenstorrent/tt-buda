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
    # pytest.skip()
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


def test_onnx_quantized_resnet(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "third_party/confidential_customer_models/model_2/onnx/quant/ResNet50-v1.5-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_ResNet50",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = DataFormat.Float32

    # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["PYBUDA_REPRODUCE_SUBGRAPH"] = "1"
    # os.environ["PYBUDA_REPRODUCE_SUBGRAPH_INPUT"] = "quantize_0.dc.buda_quantize.1"
    # os.environ["PYBUDA_REPRODUCE_SUBGRAPH_OUTPUT"] = "conv2d_1.dc.matmul.11"

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    # tti_path = "onnx_int8_resnet50_epoch_0.tti"
    # if not os.path.exists(tti_path):
    #     tt_module = pybuda_onnx_model
    #     device = pybuda.TTDevice(
    #         "tt0", module=tt_module,arch=pybuda.BackendDevice.Wormhole_B0, devtype=pybuda.BackendType.Silicon)
    #     tti_img = device.compile_to_image(
    #         img_path=tti_path,
    #         training=False,
    #         sample_inputs=[torch.randn(shape) for shape in input_shape],
    #     )


    # device_img: pybuda.TTDeviceImage = pybuda.TTDeviceImage.load_from_disk(tti_path)
    # ttdevice = pybuda.TTDevice.load_image(img=device_img)

    # inputs = [torch.randn(shape) for shape in input_shape]
    # ttdevice.push_to_inputs(*inputs)
    # output_q = pybuda.run_inference(_sequential=True)
    # output = output_q.get()[0].value().detach()

    # golden_output = pybuda_onnx_model.forward(*inputs)
    # assert np.allclose(output, golden_output[0], atol=1e-3, rtol=1e-3)
    # Compile and verify
    verify_module(
        pybuda_onnx_model,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            # verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
        ),
    )



def test_onnx_quantized_vit(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit",
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

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    # tti_path = "onnx_int8_resnet50_epoch_0.tti"
    # if not os.path.exists(tti_path):
    #     tt_module = pybuda_onnx_model
    #     device = pybuda.TTDevice(
    #         "tt0", module=tt_module,arch=pybuda.BackendDevice.Wormhole_B0, devtype=pybuda.BackendType.Silicon)
    #     tti_img = device.compile_to_image(
    #         img_path=tti_path,
    #         training=False,
    #         sample_inputs=[torch.randn(shape) for shape in input_shape],
    #     )


    # device_img: pybuda.TTDeviceImage = pybuda.TTDeviceImage.load_from_disk(tti_path)
    # ttdevice = pybuda.TTDevice.load_image(img=device_img)

    # inputs = [torch.randn(shape) for shape in input_shape]
    # ttdevice.push_to_inputs(*inputs)
    # output_q = pybuda.run_inference(_sequential=True)
    # output = output_q.get()[0].value().detach()

    # golden_output = pybuda_onnx_model.forward(*inputs)
    # assert np.allclose(output, golden_output[0], atol=1e-3, rtol=1e-3)
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
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )




def test_onnx_quantized_vit_gelu_float(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_gelu_float-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit",
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
    compiler_cfg.enable_tvm_constant_prop = True

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_encoder_gelu_float(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_1enc_gelu_float-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_1enc_gelu_float",
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
    compiler_cfg.enable_tvm_constant_prop = True

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_encoder(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_1enc-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_1enc",
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
    # compiler_cfg.place_on_new_epoch("conv2d_6.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")
    # compiler_cfg.place_on_new_epoch("conv2d_6.dc.matmul.11")
    # compiler_cfg.enable_single_buffer_fallback = True

    # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["PYBUDA_PAD_SPARSE_MM"] = "{7:8}"

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
            atol=1e-3,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_encoder_multi_output(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_1enc-extra-outputs-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_1enc_multi_output",
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
    # compiler_cfg.place_on_new_epoch("conv2d_6.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")
    # compiler_cfg.place_on_new_epoch("conv2d_6.dc.matmul.11")
    # compiler_cfg.enable_single_buffer_fallback = True

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["PYBUDA_PAD_SPARSE_MM"] = "{7:8}"

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
            atol=1e-4,
            rtol=1e-4,
            pcc=None,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_two_encoder(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_2enc-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_2enc",
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

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_seven_encoder(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_7enc-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_7enc",
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

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_four_encoder(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_4enc-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_4enc",
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
    compiler_cfg.retain_tvm_python_files = True

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"

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
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_four_encoder_zero_zp(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_4enc-Int8-zerozp.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_4enc_zerozp",
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
    compiler_cfg.retain_tvm_python_files = True

    # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"

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
            # verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_onnx_quantized_vit_full_zero_zp(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit-Int8-zerozp.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_full_zerozp",
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
    compiler_cfg.retain_tvm_python_files = True

    # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"

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
            # verify_pybuda_codegen_vs_framework=True,
            # verify_all=True,
            # atol=1e-3,
            # rtol=1e-3,
        ),
    )

def test_onnx_quantized_vit_four_encoder_softmax_zp(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_4enc-Int8-zerozp.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_4enc_softmax_zp",
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

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )



def test_onnx_quantized_vit_one_enc_sf(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit_1enc-extra-outputs-Int8-SF.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "onnx_quantized_vit_1enc_sf",
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
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            verify_all=True,
            # verify_tvm_compile=True,
            # verify_each_buda_pass=True,
        ),
    )

def test_int8_onnx_vit_calibrated(test_kind, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = "pybuda/test/quantized/simple_models/vit-Int8-calibrated.onnx"
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


    tti_path = "onnx_quantized_vit_calibrated.tti"
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

    import pickle
    file = open('data.p', 'rb')
    data = pickle.load(file)
    file.close()
    inputs = []
    labels = []
    for d in data:
        inputs.append(torch.tensor(d['input']))
        labels.append(d['label'])

    outputs = []
    for d in inputs:
        ttdevice.push_to_inputs(d)
        output_q = pybuda.run_inference(_sequential=True)
        output = output_q.get()[0].value().detach()
        outputs.append(output)

    pass_count = 0
    for i in range(len(outputs)):
        output = outputs[i]
        label = labels[i]
        print(f'label: {label}, quantized label: {np.argmax(output)}')
        if np.argmax(output) == label:
            pass_count += 1

    print("Accuracy: ", pass_count / len(outputs))

    vit_logits = {k:v for k, v in zip(labels, outputs)}

    file = open('WH_logits', 'wb')
    pickle.dump(vit_logits, file)
    file.close()
    # Compile and verify
    # verify_module(
    #     pybuda_onnx_model,
    #     input_shape,
    #     inputs=[inputs[0]],
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         test_kind=test_kind,
    #         verify_pybuda_codegen_vs_framework=True,
    #         verify_all=True,
    #         # verify_tvm_compile=True,
    #         # verify_each_buda_pass=True,
    #     ),
    # )


# def test_onnx_quantized_vit_conv(test_kind, test_device):
#     if test_device.arch == BackendDevice.Grayskull:
#         pytest.skip()

#     # Skip training
#     if test_kind.is_training():
#         pytest.skip()

#     # Download ONNX model
#     save_path = "pybuda/test/quantized/simple_models/vit_conv-Int8.onnx"
#     if not os.path.exists(save_path):
#         raise RuntimeError("Model not found")

#     # LOAD ONNX model
#     onnx_model = onnx.load(save_path)
#     onnx.checker.check_model(onnx_model)
#     pybuda_onnx_model = OnnxModule(
#         "onnx_quantized_vit_conv",
#         onnx_model,
#         save_path,
#     )
#     # Configurations
#     compiler_cfg = _get_global_compiler_config()
#     compiler_cfg.balancer_policy = "Ribbon"
#     compiler_cfg.enable_t_streaming = True
#     compiler_cfg.enable_auto_fusing = False
#     compiler_cfg.graph_solver_self_cut_type = "FastCut"
#     compiler_cfg.default_df_override = DataFormat.Float32

#     # os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
#     os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

#     # Sanity run
#     input_shape = []
#     for i in range(len(onnx_model.graph.input)):
#         dimension = onnx_model.graph.input[i].type.tensor_type.shape
#         i_shape = [d.dim_value for d in dimension.dim]
#         input_shape.append(i_shape)

#     # Compile and verify
#     verify_module(
#         pybuda_onnx_model,
#         input_shape,
#         verify_cfg=VerifyConfig(
#             arch=test_device.arch,
#             devtype=test_device.devtype,
#             test_kind=test_kind,
#             verify_pybuda_codegen_vs_framework=True,
#             verify_all=True,
#             # verify_tvm_compile=True,
#             # verify_each_buda_pass=True,
#         ),
#     )