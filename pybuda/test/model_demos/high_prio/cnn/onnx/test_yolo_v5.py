# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda, os
import requests
import torch
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from yolov5.utils.dataloaders import exif_transpose, letterbox
import onnx, pytest
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendDevice


def data_preprocessing(ims: Image.Image, size: tuple) -> tuple:
    """Data preprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : Image.Image
        Input image
    size : tuple
        Desired image size

    Returns
    -------
    tuple
        List of images, number of samples, filenames, image size, inference size, preprocessed images
    """

    _, ims = (
        (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])
    )  # number, list of images
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames

    for i, im in enumerate(ims):
        f = f"image{i}"  # filename
        im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
        files.append(Path(f).with_suffix(".jpg").name)
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = (
            im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        )  # enforce 3ch input
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = max(size) / max(s)  # gain
        shape1.append([int(y * g) for y in s])
        ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    shape1 = [size[0] for _ in np.array(shape1).max(0)]  # inf shape
    x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
    x = np.ascontiguousarray(
        np.array(x).transpose((0, 3, 1, 2))
    )  # stack and BHWC to BCHW
    x = torch.from_numpy(x) / 255  # uint8 to fp16/32
    return x


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.parametrize("variant", variants)
def test_yolo_v5_320x320_onnx(test_device, variant):

    # pybuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    input_size = 320

    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_tm_cpu_fallback = True
        if variant == "yolov5x":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"

    # Load the ONNX model
    onnx_model_path = f"third_party/confidential_customer_models/internal/yolo_v5/files/onnx/{variant}_{input_size}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    model_name = f"{variant}_{input_size}_onnx"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    pixel_values = data_preprocessing(image, size=(input_size, input_size))

    # Run inference on Tenstorrent device
    verify_module(
        pybuda.OnnxModule(model_name, onnx_model, onnx_model_path),
        input_shapes=([pixel_values.shape]),
        inputs=([pixel_values]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.parametrize("variant", variants)
def test_yolo_v5_480x480_onnx(test_device, variant):

    # pybuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_tm_cpu_fallback = True
    # Temp mitigations for net2pipe errors, should be removed.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

    if test_device.arch == BackendDevice.Wormhole_B0:

        os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
        os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
        if variant == "yolov5m":
            compiler_cfg.balancer_op_override(
                "concatenate_19.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                "grid_shape",
                (1, 1),
            )
            compiler_cfg.balancer_op_override(
                "concatenate_26.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                "grid_shape",
                (1, 1),
            )
            compiler_cfg.place_on_new_epoch("concatenate_26.dc.concatenate.30.dc.concatenate.1.dc.buffer.0")
        elif variant == "yolov5s":
            compiler_cfg.balancer_op_override(
                "concatenate_19.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                "grid_shape",
                (1, 1),
            )
            compiler_cfg.balancer_op_override(
                "concatenate_26.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                "grid_shape",
                (1, 1),
            )
        elif variant == "yolov5n":
            compiler_cfg.balancer_op_override(
                "concatenate_19.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                "t_stream_shape",
                (1, 1),
            )
            os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"
        elif variant == "yolov5x":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    elif test_device.arch == BackendDevice.Grayskull:

        if variant in ["yolov5n", "yolov5s"]:
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        if variant in ["yolov5m", "yolov5x"]:
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"
            if variant == "yolov5m":
                compiler_cfg.balancer_op_override(
                    "concatenate_26.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                    "grid_shape",
                    (1, 1),
                )
            if variant == "yolov5x":
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
                compiler_cfg.balancer_op_override(
                    "concatenate_40.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                    "grid_shape",
                    (1, 1),
                )

    elif test_device.arch == BackendDevice.Blackhole:

        if variant == "yolov5n":
            compiler_cfg.place_on_new_epoch("_fused_op_7")
        elif variant == "yolov5l":
            compiler_cfg.place_on_new_epoch("_fused_op_11")
            compiler_cfg.place_on_new_epoch("_fused_op_12")
            compiler_cfg.place_on_new_epoch("_fused_op_25")
        elif variant == "yolov5x":
            compiler_cfg.place_on_new_epoch("conv2d_44.dc.matmul.11")
            compiler_cfg.place_on_new_epoch("_fused_op_13")

    input_size = 480

    # Load the ONNX model
    onnx_model_path = f"third_party/confidential_customer_models/internal/yolo_v5/files/onnx/{variant}_{input_size}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    model_name = f"{variant}_{input_size}_onnx"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    pixel_values = data_preprocessing(image, size=(input_size, input_size))

    # Run inference on Tenstorrent device
    verify_module(
        pybuda.OnnxModule(model_name, onnx_model, onnx_model_path),
        input_shapes=([pixel_values.shape]),
        inputs=([pixel_values]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.parametrize("variant", variants)
def test_yolo_v5_640x640_onnx(test_device, variant):

    # pybuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Wormhole_B0:

        os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
        os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"

        if variant in ["yolov5n", "yolov5s"]:
            if variant == "yolov5s":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
            compiler_cfg.balancer_op_override(
                "concatenate_259.dc.concatenate.7", "grid_shape", (1, 1)
            )

        if variant == "yolov5m":
            compiler_cfg.balancer_op_override(
                "concatenate_332.dc.concatenate.7", "grid_shape", (1, 1)
            )
            compiler_cfg.balancer_op_override(
                "concatenate_332.dc.concatenate.7", "t_stream_shape", (1, 1)
            )
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"

        if variant == "yolov5l":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
            compiler_cfg.balancer_op_override(
                "concatenate_405.dc.concatenate.7", "grid_shape", (1, 1)
            )

        if variant == "yolov5x":
            compiler_cfg.balancer_op_override(
                "concatenate_478.dc.concatenate.7", "grid_shape", (1, 1)
            )
            compiler_cfg.enable_auto_fusing = False
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "382976"
            compiler_cfg.place_on_new_epoch("concatenate_40.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12")
            compiler_cfg.place_on_new_epoch("concatenate_478.dc.sparse_matmul.10.lc2")


    elif test_device.arch == BackendDevice.Grayskull:

        compiler_cfg.enable_tm_cpu_fallback = True

        if variant=="yolov5n":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"

        if variant == "yolov5l":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"

        if variant in ["yolov5m", "yolov5x"]:
            os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
            os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"

            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

            if variant == "yolov5m":
                compiler_cfg.balancer_op_override(
                    "concatenate_26.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                    "grid_shape",
                    (1, 1),
                )
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{150*1024}"

            if variant == "yolov5x":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
                compiler_cfg.balancer_op_override(
                    "concatenate_40.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12",
                    "grid_shape",
                    (1, 1),
                )

    elif test_device.arch == BackendDevice.Blackhole:

        os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
        os.environ["PYBUDA_CONCAT_SLICE_Y"] = "10"

        if variant == "yolov5n":
            compiler_cfg.balancer_op_override("concatenate_259.dc.concatenate.7", "grid_shape", (1, 1))

        elif variant == "yolov5s":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
            compiler_cfg.balancer_op_override("concatenate_259.dc.concatenate.7", "grid_shape", (1, 1))

        elif variant == "yolov5m":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
            compiler_cfg.balancer_op_override("concatenate_332.dc.concatenate.7", "grid_shape", (1, 1))
            compiler_cfg.balancer_op_override("concatenate_332.dc.concatenate.7", "t_stream_shape", (1, 1))

        elif variant == "yolov5l":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{112*1024}"
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
            compiler_cfg.balancer_op_override("concatenate_405.dc.concatenate.7", "grid_shape", (1, 1))

        elif variant == "yolov5x":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{374*1024}"
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.place_on_new_epoch("concatenate_40.dc.concatenate.30.dc.concatenate.0.dc.concatenate.12")
            compiler_cfg.place_on_new_epoch("concatenate_478.dc.sparse_matmul.10.lc2")
            compiler_cfg.balancer_op_override("concatenate_478.dc.concatenate.7", "grid_shape", (1, 1))

    input_size = 640

    # Load the ONNX model
    onnx_model_path = f"third_party/confidential_customer_models/internal/yolo_v5/files/onnx/{variant}_{input_size}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    model_name = f"{variant}_{input_size}_onnx"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    pixel_values = data_preprocessing(image, size=(input_size, input_size))

    # Run inference on Tenstorrent device
    verify_module(
        pybuda.OnnxModule(model_name, onnx_model, onnx_model_path),
        input_shapes=([pixel_values.shape]),
        inputs=([pixel_values]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
