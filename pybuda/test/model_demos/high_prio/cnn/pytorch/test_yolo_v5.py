# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from test.utils import download_model
import torch
import torch.nn as nn
from PIL import Image

import pybuda

from pybuda import (
    PyTorchModule,
    VerifyConfig,
    DataFormat
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(test_device, variant, size):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tm_cpu_fallback = False
    compiler_cfg.enable_conv_prestride = True
    compiler_cfg.enable_tvm_constant_prop = True
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_tm_cpu_fallback = True
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
    elif test_device.arch == BackendDevice.Wormhole_B0:
        if size == "m":
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        os.environ["PYBUDA_RIBBON2"] = "1"
        compiler_cfg.default_df_override = DataFormat.Float16_b
        os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16, 3:4}"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = f"{64*1024}"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{6*1024}"
        if size == "l" or size == "x":
            compiler_cfg.enable_enumerate_u_kt = False
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
            os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
        if size == "l" or size == "m" or size == "x":
            compiler_cfg.enable_auto_fusing = False

    name = "yolov5" + size

    model = download_model(torch.hub.load, variant, name, pretrained=True)
    module = PyTorchModule("pt_" + name + "_320x320", model)

    input_shape = (1, 3, 320, 320)
    
    return module, [input_shape], {}


size = ["n", "s", "m", "l", "x"]
@pytest.mark.parametrize(
    "size", size, ids=["yolov5" + s for s in size]
)
def test_yolov5_320x320(test_device, size):
    model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        test_device, "ultralytics/yolov5",
        size=size,
    )

    verify_module(
        model,
        (inputs[0],),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework = True,
        ),
    )


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(test_device, variant, size):
    # Add required env vars as per: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/model-demos/-/issues/46
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_DISABLE_CAP_SPARSE_MM_FIDELITY"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_enumerate_u_kt = False
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        compiler_cfg.enable_tm_cpu_fallback = True
        compiler_cfg.enable_conv_prestride = True
        os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16, 3:4}"
        if size in ["s", "m", "l", "x", "n"]:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{65*1024}"
        if size in ["l", "x"]:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"
            os.environ[
                "PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"
            ] = "FastCut"
            compiler_cfg.enable_enumerate_u_kt = True
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            os.environ["PYBUDA_RIBBON2"] = "1"
            if size in ["x"]:
                compiler_cfg.place_on_new_epoch("conv2d_210.dc.matmul.11")
                os.environ["PYBUDA_TEMP_BALANCER_DISABLE_TARGET_PROXIMITY"] = "1"
                os.environ["PYBUDA_TEMP_RIBBON2_LEGACY_UTIL_EVAL"] = "1"
        if size in ["m"]:
            os.environ["PYBUDA_RIBBON2"] = "1"
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            os.environ["PYBUDA_TEMP_BALANCER_DISABLE_TARGET_PROXIMITY"] = "1"
            compiler_cfg.place_on_new_epoch("conv2d_27.dc.matmul.8")
        if size in ["l"]:
            compiler_cfg.place_on_new_epoch("conv2d_313.dc.matmul.8")


    elif test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16, 3:4}"
        os.environ["PYBUDA_MAX_GRAPH_CUT_RETRY"] = "100"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
        os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
        os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "0"
        if size == "s" or size == "m" or size == "l" or size == "x":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        compiler_cfg.enable_conv_prestride = True
        compiler_cfg.enable_tvm_constant_prop = True
        compiler_cfg.default_df_override = DataFormat.Float16_b
        if size == "n" or size == "m":
            compiler_cfg.enable_tm_cpu_fallback = False
        if size == "s" or size == "n" or size == "l":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
            compiler_cfg.balancer_op_override("concatenate_259.dc.concatenate.7", "grid_shape", (1,1))
        if size == "n":
            compiler_cfg.balancer_op_override("concatenate_19.dc.concatenate.30.dc.concatenate.1.dc.buffer.0", "t_stream_shape", (3,1))
        if size == "m":
            compiler_cfg.balancer_op_override("concatenate_332.dc.concatenate.7", "grid_shape", (1,1))
            os.environ["PYBUDA_RIBBON2"] = "1"
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{112*1024}"
            os.environ["PYBUDA_TEMP_RIBBON2_LEGACY_UTIL_EVAL"] = "1"
        if size == "l":
            compiler_cfg.enable_auto_transposing_placement = True
            compiler_cfg.enable_tm_cpu_fallback = True
            os.environ["PYBUDA_RIBBON2"] = "1"
            compiler_cfg.balancer_op_override("conv2d_328.dc.matmul.8", "grid_shape", (5,2))
        if size == "x":
            compiler_cfg.balancer_op_override("concatenate_363.dc.concatenate.0", "grid_shape", (1,1))
            compiler_cfg.balancer_op_override("conv2d_41.dc.matmul.8", "t_stream_shape", (1,1))
            os.environ["PYBUDA_RIBBON2"] = "1"
            compiler_cfg.enable_tm_cpu_fallback = True
            os.environ["PYBUDA_DISABLE_CAP_SPARSE_MM_FIDELITY"] = "0"
            os.environ["PYBUDA_TEMP_BALANCER_DISABLE_TARGET_PROXIMITY"] = "1"

    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)
    module = PyTorchModule("pt_" + name + "_640x640", model)

    input_shape = (1, 3, 640, 640)

    return module, [input_shape], {}


size = ["n", "s", "m", "l", "x"]
@pytest.mark.parametrize(
    "size", size, ids=["yolov5" + s for s in size]
)
def test_yolov5_640x640(test_device, size):
    if size in ["yolov5x", "yolov5l"]:
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    model, inputs, _ = generate_model_yoloV5I640_imgcls_torchhub_pytorch(
        test_device, "ultralytics/yolov5",
        size=size,
    )

    verify_module(
        model,
        (inputs[0],),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework = True,
        ),
    )


def generate_model_yoloV5I480_imgcls_torchhub_pytorch(test_device, variant, size):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tm_cpu_fallback = True
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_PAD_SPARSE_MM"] = "{113:128}"
        if size == "x":
            os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            compiler_cfg.balancer_op_override("concatenate_40.dc.concatenate.30.dc.concatenate.1.dc.buffer.0", "t_stream_shape", (6,1))
            compiler_cfg.balancer_op_override("conv2d_41.dc.matmul.8", "grid_shape", (5,5))
        elif size == "m":
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            compiler_cfg.balancer_op_override("concatenate_26.dc.concatenate.30.dc.concatenate.1.dc.buffer.0", "t_stream_shape", (6,1))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{32*1024}"
        else:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{16*1024}"

    elif test_device.arch == BackendDevice.Wormhole_B0:
        # Add required env vars as per: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/model-demos/-/issues/46
        os.environ["PYBUDA_RIBBON2"] = "1"
        os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16, 3:4}"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"]  = f"{64*1024}"
        compiler_cfg.default_df_override = DataFormat.Float16_b
        
        if size == "s":
            compiler_cfg.default_dram_parameters = False
        else:
            compiler_cfg.default_dram_parameters = True

        if size == "m":
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            compiler_cfg.balancer_op_override("concatenate_26.dc.concatenate.30.dc.concatenate.1.dc.buffer.0", "t_stream_shape", (6,1))
        elif size == "l":
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.place_on_new_epoch("concatenate_208.dc.concatenate.0")
        elif size == "x":
            compiler_cfg.enable_auto_fusing = False
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)
    module = PyTorchModule("pt_" + name + "_480x480", model)

    input_shape = (1, 3, 480, 480)
    
    return module, [input_shape], {}


@pytest.mark.parametrize(
    "size", size, ids=["yolov5" + s for s in size]
)
def test_yolov5_480x480(test_device, size):
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
    if size in ["yolov5x", "yolov5l", "yolov5m"] and test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    model, inputs, _ = generate_model_yoloV5I480_imgcls_torchhub_pytorch(
        test_device, "ultralytics/yolov5",
        size=size,
    )

    verify_module(
        model,
        (inputs[0],),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework = True,
            verify_post_placer=False
        ),
    )


@pytest.mark.skip(reason="Not supported")
def test_yolov5_1280x1280(test_device):
    # Add required env vars as per: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/model-demos/-/issues/46
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16}"
    os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"

    os.environ["PYBUDA_PADDING_PASS_BUFFER_QUEUE"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_tm_cpu_fallback = True
    compiler_cfg.default_df_override = DataFormat.Float16_b
    compiler_cfg.paddings = {
        "concatenate_19.dc.concatenate.4": True,
        "concatenate_46.dc.concatenate.4": True,
        "concatenate_139.dc.concatenate.4": True,
        "concatenate_152.dc.concatenate.4": True
    }

    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5s", pretrained=True)
    module = PyTorchModule("pt_yolov5s", model)

    input_shape = (1, 3, 1280, 1280)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework = True,
        ),
    )
