# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pybuda
from PIL import Image
import requests
from torchvision import transforms
import os
import pytest
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda import VerifyConfig
import sys

sys.path.append("third_party/confidential_customer_models/cv_demos/retinanet/model/")
from model_implementation import Model
from pybuda._C.backend_api import BackendDevice
import zipfile
import shutil


def img_preprocess():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img


variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn101fpn",
    "retinanet_rn152fpn",
]


@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant, test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "73728"

    if test_device.arch == BackendDevice.Wormhole_B0:

        if variant == "retinanet_rn18fpn":
            compiler_cfg.place_on_new_epoch("conv2d_357.dc.matmul.11")
            compiler_cfg.balancer_op_override(
                "conv2d_322.dc.matmul.11", "t_stream_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_300.dc.matmul.11", "grid_shape", (1, 1)
            )

        elif variant == "retinanet_rn34fpn":
            compiler_cfg.place_on_new_epoch("conv2d_589.dc.matmul.11")
            compiler_cfg.balancer_op_override(
                "conv2d_554.dc.matmul.11", "t_stream_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_532.dc.matmul.11", "grid_shape", (1, 1)
            )

        elif variant == "retinanet_rn50fpn":
            compiler_cfg.place_on_new_epoch("conv2d_826.dc.matmul.11")
            compiler_cfg.balancer_op_override(
                "conv2d_791.dc.matmul.11", "t_stream_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_769.dc.matmul.11", "grid_shape", (1, 1)
            )

        elif variant == "retinanet_rn101fpn":
            compiler_cfg.place_on_new_epoch("conv2d_1557.dc.matmul.11")
            compiler_cfg.balancer_op_override(
                "conv2d_1522.dc.matmul.11", "t_stream_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1500.dc.matmul.11", "grid_shape", (1, 1)
            )

        elif variant == "retinanet_rn152fpn":
            compiler_cfg.place_on_new_epoch("conv2d_2288.dc.matmul.11")
            compiler_cfg.balancer_op_override(
                "conv2d_2253.dc.matmul.11", "t_stream_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_2231.dc.matmul.11", "grid_shape", (1, 1)
            )

    if test_device.arch == BackendDevice.Grayskull:
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

        if variant == "retinanet_rn18fpn":
            compiler_cfg.balancer_op_override(
                "conv2d_322.dc.matmul.11", "t_stream_shape", (1, 1)
            )

        elif variant == "retinanet_rn34fpn":
            compiler_cfg.balancer_op_override(
                "conv2d_554.dc.matmul.11", "t_stream_shape", (1, 1)
            )

        elif variant == "retinanet_rn50fpn":
            compiler_cfg.balancer_op_override(
                "conv2d_791.dc.matmul.11", "t_stream_shape", (1, 1)
            )

        elif variant == "retinanet_rn101fpn":
            compiler_cfg.balancer_op_override(
                "conv2d_1522.dc.matmul.11", "t_stream_shape", (1, 1)
            )

        elif variant == "retinanet_rn152fpn":
            compiler_cfg.balancer_op_override(
                "conv2d_2253.dc.matmul.11", "t_stream_shape", (1, 1)
            )

    # Prepare model
    url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant}.zip"
    local_zip_path = f"{variant}.zip"

    response = requests.get(url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Find the path of the .pth file
    extracted_path = f"{variant}"
    checkpoint_path = ""
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)

    model = Model.load(checkpoint_path)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)

    # Prepare input
    input_batch = img_preprocess()

    # Inference
    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )

    # Delete the extracted folder and the zip file
    shutil.rmtree(extracted_path)
    os.remove(local_zip_path)
