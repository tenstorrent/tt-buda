# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import subprocess

subprocess.run(["pip", "install", "yolox==0.3.0", "--no-deps"])  # Install yolox==0.3.0 without installing its dependencies

"""
Reason to install yolox=0.3.0 through subprocess : 
requirements of yolox=0.3.0 can be found here https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/requirements.txt
onnx==1.8.1 and onnxruntime==1.8.0 are required by yolox which are incompatible with our package versions
Dependencies required by yolox for pytorch implemetation are already present in pybuda and packages related to onnx is not needed 
pip install yolox==0.3.0 --no-deps can be used to install a package without installing its dependencies through terminal
But in pybuda packages were installed through requirements.txt file not though terminal. 
unfortunately there is no way to include --no-deps in  requirements.txt file.
for this reason , yolox==0.3.0 is intalled through subprocess.
"""

import pybuda
import torch
import cv2
import numpy as np
from yolox.exp import get_exp
import requests
import pytest
import os
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendDevice


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = torch.from_numpy(padded_img)
    return padded_img


variants = ["yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]


@pytest.mark.parametrize("variant", variants)
def test_yolox_pytorch(variant, test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    if test_device.arch == BackendDevice.Wormhole_B0:

        if variant in ["yolox_nano", "yolox_tiny"]:
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"

        elif variant == "yolox_s":
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_33.dc.matmul.8", "t_stream_shape", (1, 1))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"
            compiler_cfg.place_on_new_epoch("concatenate_1163.dc.sparse_matmul.11.lc2")

        elif variant in ["yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]:
            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

            if variant == "yolox_m":
                compiler_cfg.place_on_new_epoch("conv2d_811.dc.matmul.8")
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 6))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (5, 1))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_1530.dc.sparse_matmul.11.lc2")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"

            elif variant == "yolox_l":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "12288"
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 12))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_1897.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_darknet":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "53248"
                compiler_cfg.place_on_new_epoch("concatenate_1242.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_x":
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (5, 1))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_2264.dc.sparse_matmul.11.lc2")

    elif test_device.arch == BackendDevice.Grayskull:

        if variant in ["yolox_nano", "yolox_tiny"]:
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 1))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13))

        elif variant == "yolox_s":
            compiler_cfg.balancer_op_override("concatenate_1163.dc.concatenate.7.before_padded_node.nop_0", "grid_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5))
            compiler_cfg.place_on_new_epoch("concatenate_1163.dc.sparse_matmul.11.lc2")

        elif variant in ["yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]:
            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

            if variant == "yolox_m":
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5))
                compiler_cfg.place_on_new_epoch("concatenate_1530.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_l":
                compiler_cfg.place_on_new_epoch("conv2d_1410.dc.conv2d.1.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1644.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.place_on_new_epoch("concatenate_1897.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_darknet":
                compiler_cfg.place_on_new_epoch("conv2d_1070.dc.conv2d.3.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1070.dc.conv2d.5.dc.matmul.11")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "12288"
                compiler_cfg.place_on_new_epoch("conv2d_1070.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")
                compiler_cfg.place_on_new_epoch("conv2d_1070.dc.conv2d.1.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1147.dc.matmul.11")
                compiler_cfg.balancer_op_override("concatenate_1242.dc.concatenate.7.before_padded_node.nop_0", "grid_shape", (1, 1))

            elif variant == "yolox_x":
                compiler_cfg.place_on_new_epoch("conv2d_1717.dc.conv2d.5.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1717.dc.conv2d.1.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1717.dc.conv2d.3.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1699.dc.matmul.8")
                compiler_cfg.place_on_new_epoch("conv2d_1732.dc.matmul.8")
                compiler_cfg.place_on_new_epoch("conv2d_1981.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_1736.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.place_on_new_epoch("concatenate_2264.dc.sparse_matmul.11.lc2")

    # prepare model
    weight_name = f"{variant}.pth"
    url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{weight_name}"
    response = requests.get(url)
    with open(f"{weight_name}", "wb") as file:
        file.write(response.content)

    if variant == "yolox_darknet":
        model_name = "yolov3"
    else:
        model_name = variant.replace("_", "-")

    exp = get_exp(exp_name=model_name)
    model = exp.get_model()
    ckpt = torch.load(f"{variant}.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model_name = f"pt_{variant}"
    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)

    # prepare input
    if variant in ["yolox_nano", "yolox_tiny"]:
        input_shape = (416, 416)
    else:
        input_shape = (640, 640)

    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    response = requests.get(url)
    with open("input.jpg", "wb") as f:
        f.write(response.content)
    img = cv2.imread("input.jpg")
    img_tensor = preprocess(img, input_shape)
    img_tensor = img_tensor.unsqueeze(0)

    # pcc
    pcc_value = 0.99

    if test_device.arch == BackendDevice.Wormhole_B0 and variant == "yolox_nano":
        pcc_value = 0.96

    elif test_device.arch == BackendDevice.Grayskull:
        if variant in ["yolox_nano", "yolox_s", "yolox_l", "yolox_x"]:
            pcc_value = 0.93
        elif variant in ["yolox_m,yolox_darknet"]:
            pcc_value = 0.92
        elif variant == "yolox_tiny":
            pcc_value = 0.98

    # Inference
    verify_module(
        tt_model,
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc_value,
        ),
    )

    # remove downloaded weights,image
    os.remove(weight_name)
    os.remove("input.jpg")
