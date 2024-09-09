# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import onnx

import os
import requests
from PIL import Image

from transformers import AutoImageProcessor

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


def test_perceiverio_conv_imgcls_onnx(test_device):

    model_name = "deepmind/vision-perceiver-conv"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    verify_enabled = True

    pcc_value = 0.96
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{10*1024}"
        compiler_cfg.balancer_op_override("multiply_19", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_142", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_3103", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_3123", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_2745", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_2934", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_79", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override("multiply_99", "t_stream_shape", (1, 1))
        compiler_cfg.balancer_op_override(
            "max_pool2d_35.dc.reshape.10.dc.sparse_matmul.13.lc2",
            "t_stream_shape",
            (1, 1),
        )

    elif test_device.arch == pybuda.BackendDevice.Grayskull:

        if test_device.devtype == pybuda.BackendType.Silicon:
            verify_enabled = False

    onnx_model_path = (
        "third_party/confidential_customer_models/internal/perceiverio/files/onnx/"
        + str(model_name).split("/")[-1].replace("-", "_")
        + ".onnx"
    )

    # Sample Image
    pixel_values = get_sample_data(model_name)

    # Load the onnx model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Create PyBuda module from Onnx model
    tt_model = pybuda.OnnxModule(
        str(model_name.split("/")[-1].replace("-", "_")) + "_onnx",
        onnx_model,
        onnx_model_path,
    )

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=(pixel_values.shape,),
        inputs=[(pixel_values,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=verify_enabled,  # pcc drops in silicon devicetype
            pcc=pcc_value,
        ),
    )
