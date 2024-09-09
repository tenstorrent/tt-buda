import pybuda, os
import pytest
import cv2, torch
import numpy as np
import onnx
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
import requests
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
def test_yolox_onnx(variant, test_device):

    # pybuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    if test_device.arch == BackendDevice.Wormhole_B0:

        if variant in ["yolox_nano", "yolox_tiny"]:

            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"

        elif variant == "yolox_s":

            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_33.dc.matmul.8", "t_stream_shape", (1, 1))
            compiler_cfg.place_on_new_epoch("concatenate_275.dc.sparse_matmul.11.lc2")

        elif variant == "yolox_m":

            os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
            os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
            os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"

            compiler_cfg.place_on_new_epoch("conv2d_187.dc.matmul.8")
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (5, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.place_on_new_epoch("concatenate_354.dc.sparse_matmul.11.lc2")

        elif variant in ["yolox_l", "yolox_darknet", "yolox_x"]:

            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

            if variant == "yolox_l":

                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_433.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_darknet":

                compiler_cfg.balancer_op_override("conv2d_28.dc.matmul.8", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_33.dc.matmul.8", "t_stream_shape", (1, 1))
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "53248"
                compiler_cfg.place_on_new_epoch("concatenate_222.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_x":

                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_512.dc.sparse_matmul.11.lc2")

    elif test_device.arch == BackendDevice.Grayskull:

        if variant in ["yolox_nano", "yolox_tiny"]:

            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (8, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (8, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (8, 1))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.14.lc2", "grid_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.14.lc2", "grid_shape", (1, 8))

        elif variant == "yolox_s":

            compiler_cfg.balancer_op_override(
                "concatenate_275.dc.concatenate.7_to_concatenate_275.dc.sparse_matmul.11.lc2_1_serialized_dram_queue.before_padded_node.nop_0",
                "grid_shape",
                (1, 1),
            )
            compiler_cfg.balancer_op_override("concatenate_275.dc.concatenate.7.before_padded_node.nop_0", "grid_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (8, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
            compiler_cfg.place_on_new_epoch("concatenate_275.dc.sparse_matmul.11.lc2")
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"

        elif variant == "yolox_m":

            compiler_cfg.balancer_op_override("concatenate_354.dc.concatenate.7.before_padded_node.nop_0", "grid_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
            compiler_cfg.place_on_new_epoch("concatenate_354.dc.sparse_matmul.11.lc2")

        elif variant in ["yolox_l", "yolox_darknet", "yolox_x"]:

            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

            if variant == "yolox_l":
                os.environ["PYBUDA_RIBBON2_CONSERVATIVE_OPTIMIZATION_ITERATIONS"] = "0"
                compiler_cfg.place_on_new_epoch("conv2d_372.dc.matmul.11")
                compiler_cfg.balancer_op_override("concatenate_433.dc.concatenate.7.before_padded_node.nop_0", "grid_shape", (1, 1))
                compiler_cfg.balancer_op_override(
                    "concatenate_433.dc.concatenate.7_to_concatenate_433.dc.sparse_matmul.11.lc2_1_serialized_dram_queue.before_padded_node.nop_0",
                    "grid_shape",
                    (1, 1),
                )
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 60))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (5, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.place_on_new_epoch("concatenate_433.dc.sparse_matmul.11.lc2")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"
                compiler_cfg.place_on_new_epoch("_fused_op_163")

            elif variant == "yolox_darknet":

                compiler_cfg.place_on_new_epoch("_fused_op_34")
                compiler_cfg.place_on_new_epoch("conv2d_199.dc.matmul.11")
                compiler_cfg.balancer_op_override("concatenate_222.dc.concatenate.7.before_padded_node.nop_0", "grid_shape", (1, 1))
                compiler_cfg.place_on_new_epoch("concatenate_222.dc.sparse_matmul.11.lc2")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"

            elif variant == "yolox_x":

                os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
                os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
                os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"

                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                compiler_cfg.place_on_new_epoch("conv2d_385.dc.conv2d.5.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_385.dc.conv2d.1.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("conv2d_385.dc.conv2d.3.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 8))
                compiler_cfg.place_on_new_epoch("concatenate_512.dc.sparse_matmul.11.lc2")

    if test_device.arch == BackendDevice.Blackhole:

        if variant in ["yolox_nano", "yolox_tiny"]:

            compiler_cfg.place_on_new_epoch("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.14.lc2")
            compiler_cfg.place_on_new_epoch("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.14.lc2")
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"

        elif variant == "yolox_s":

            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_33.dc.matmul.8", "t_stream_shape", (1, 1))
            compiler_cfg.place_on_new_epoch("concatenate_275.dc.sparse_matmul.11.lc2")

        elif variant == "yolox_m":

            os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
            os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
            os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"

            compiler_cfg.place_on_new_epoch("conv2d_187.dc.matmul.8")
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (5, 1))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
            compiler_cfg.place_on_new_epoch("concatenate_354.dc.sparse_matmul.11.lc2")

        elif variant in ["yolox_l", "yolox_darknet", "yolox_x"]:

            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

            if variant == "yolox_l":

                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_433.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_darknet":

                compiler_cfg.balancer_op_override("conv2d_28.dc.matmul.8", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_33.dc.matmul.8", "t_stream_shape", (1, 1))
                compiler_cfg.place_on_new_epoch("concatenate_222.dc.sparse_matmul.11.lc2")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "53248"

            elif variant == "yolox_x":

                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.balancer_op_override("conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4))
                compiler_cfg.place_on_new_epoch("concatenate_512.dc.sparse_matmul.11.lc2")
                compiler_cfg.place_on_new_epoch("conv2d_379.dc.matmul.8")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"

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

    # Load and validate the ONNX model
    onnx_model_path = f"third_party/confidential_customer_models/internal/yolox/files/onnx/{variant}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    model_name = f"onnx_{variant}"
    tt_model = pybuda.OnnxModule(model_name, onnx_model, onnx_model_path)

    # PCC
    if test_device.arch in [BackendDevice.Wormhole_B0, BackendDevice.Blackhole]:
        if variant == "yolox_nano":
            pcc = 0.93
        else:
            pcc = 0.99
    elif test_device.arch == BackendDevice.Grayskull:
        if variant == "yolox_nano":
            pcc = 0.91
        elif variant in ["yolox_m", "yolox_darknet"]:
            pcc = 0.92
        elif variant in ["yolox_s", "yolox_l"]:
            pcc = 0.93
        elif variant == "yolox_x":
            pcc = 0.94
        else:
            pcc = 0.99

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
            pcc=pcc,
        ),
    )
