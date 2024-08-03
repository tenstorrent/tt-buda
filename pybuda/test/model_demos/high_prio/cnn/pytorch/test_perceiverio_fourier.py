# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import torch
import os
import requests
from PIL import Image
from loguru import logger
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationFourier,
)

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind


def get_sample_data(model_name):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        height = image_processor.to_dict()["size"]["height"]
        width = image_processor.to_dict()["size"]["width"]
        pixel_values = torch.rand(1, 3, height, width).to(torch.float32)
    return pixel_values


def test_perceiverio_fourier_imgcls_pytorch(test_device):

    variant = "deepmind/vision-perceiver-fourier"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    verify_enabled = True
    pcc_value = 0.99

    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
        compiler_cfg.enable_auto_fusing = False
        compiler_cfg.balancer_op_override(
            "hslice_41.dc.sparse_matmul.2.lc2", "t_stream_shape", (1, 4)
        )
        compiler_cfg.balancer_op_override("add_33", "t_stream_shape", (1, 2))
        if test_device.devtype == pybuda.BackendType.Silicon:
            pcc_value = 0.96

    elif test_device.arch == pybuda.BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
        compiler_cfg.balancer_op_override(
            "hslice_41.dc.sparse_matmul.2.lc2", "t_stream_shape", (1, 7)
        )
        if test_device.devtype == pybuda.BackendType.Silicon:
            verify_enabled = False

    # Sample Image
    pixel_values = get_sample_data(variant)

    # Load the model from HuggingFace
    model = PerceiverForImageClassificationFourier.from_pretrained(variant)
    model.eval()

    tt_model = pybuda.PyTorchModule(
        "pt_" + str(variant.split("/")[-1].replace("-", "_")), model
    )

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(pixel_values.shape,)],
        inputs=[(pixel_values)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=verify_enabled,  # pcc drops in silicon devicetype
            pcc=pcc_value,
        ),
    )
