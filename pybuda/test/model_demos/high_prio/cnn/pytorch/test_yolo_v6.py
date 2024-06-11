# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import os
import pytest
import requests
import math, cv2
import numpy as np
import torch
from PIL import Image
from yolov6 import YOLOV6
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendDevice

# preprocessing steps referred form https://github.com/meituan/YOLOv6/blob/main/inference.ipynb


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
        new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return im, r, (left, top)


def check_img_size(img_size, s=32, floor=0):
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(
            f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(path, img_size, stride, half):
    """Process image before image inference."""

    img_src = np.asarray(Image.open(requests.get(path, stream=True).raw))
    image = letterbox(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


# Didn't dealt with yolov6n6,yolov6s6,yolov6m6,yolov6l6 variants because of its higher input size(1280)
variants = ["yolov6n", "yolov6s", "yolov6m", "yolov6l"]


@pytest.mark.parametrize("variant", variants)
def test_yolo_v6_pytorch(variant, test_device):

    # STEP 1 : Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if variant in ["yolov6m", "yolov6l"]:
        os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
        os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        os.environ["PYBUDA_MAX_FORK_JOIN_BUF"] = "1"

        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

        if test_device.arch == BackendDevice.Grayskull and variant == "yolov6m":
            compiler_cfg.balancer_op_override(
                "conv2d_258.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_258.dc.reshape.12.dc.sparse_matmul.3.lc2",
                "t_stream_shape",
                (2, 1),
            )

        if test_device.arch == BackendDevice.Wormhole_B0 and variant == "yolov6l":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        if test_device.arch == BackendDevice.Grayskull and variant == "yolov6l":
            compiler_cfg.balancer_op_override(
                "conv2d_484.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "conv2d_484.dc.reshape.12.dc.sparse_matmul.3.lc2",
                "t_stream_shape",
                (2, 1),
            )

    # STEP 2 :prepare model
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{variant}.pt"
    weights = f"{variant}.pt"

    try:
        response = requests.get(url)
        with open(weights, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {url} to {weights}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

    model = YOLOV6(weights)
    model = model.model
    model.eval()

    tt_model = pybuda.PyTorchModule(f"{variant}_pt", model)

    # STEP 3 : prepare input
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    stride = 32
    input_size = 640
    img_size = check_img_size(input_size, s=stride)
    img, img_src = process_image(url, img_size, stride, half=False)
    input_batch = img.unsqueeze(0)

    # STEP 4 : Inference
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

    # STEP 5 : remove downloaded weights
    os.remove(weights)
