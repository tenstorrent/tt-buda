# YOLOv5 Demo - PyTorch

import os
import sys

import pybuda
import requests
import torch
from cv_demos.yolo_v5.utils.processing import data_postprocessing, data_preprocessing
from PIL import Image
from pybuda._C.backend_api import BackendDevice


def run_pytorch_yolov5_320(variant="yolov5s"):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_tm_cpu_fallback = False
    compiler_cfg.enable_conv_prestride = True
    compiler_cfg.enable_tvm_constant_prop = True
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    if variant == "yolov5m":
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

    # Load YOLOv5 model
    # Variants: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
    model_ckpt = variant

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            # Set PyBUDA environment variables
            compiler_cfg.enable_tm_cpu_fallback = True
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        elif available_devices[0] == BackendDevice.Wormhole_B0:
            if variant == "yolov5m":
                os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
            os.environ["PYBUDA_RIBBON2"] = "1"
            compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16, 3:4}"
            os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
            os.environ["PYBUDA_EXTRA_L1_MARGIN"] = str(64 * 1024)
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{6*1024}"
            if model_ckpt in ["yolov5l", "yolov5x"]:
                compiler_cfg.enable_enumerate_u_kt = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
                compiler_cfg.enable_auto_fusing = False
            elif model_ckpt in ["yolov5n", "yolov5s"]:
                compiler_cfg.enable_auto_fusing = False
        else:
            print("not a supported device!")
            sys.exit()

    # NOTE: Can alternatively load models from yolov5 package
    # import yolov5; model = yolov5.load("yolov5s.pt")
    model = torch.hub.load("ultralytics/yolov5", model_ckpt, device="cpu")

    # Set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    pixel_size = 320  # image pixel size

    # Load data sample
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    ims, n, files, shape0, shape1, pixel_values = data_preprocessing(image, size=(pixel_size, pixel_size))

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule(f"pt_{model_ckpt}_{pixel_size}", model),
        inputs=([pixel_values]),
        _verify_cfg=pybuda.verify.VerifyConfig(verify_pybuda_codegen_vs_framework=True),
    )
    output = output_q.get()

    # Data postprocessing on Host
    results = data_postprocessing(
        ims,
        pixel_values.shape,
        output[0].value(),
        model,
        n,
        shape0,
        shape1,
        files,
    )

    # Print results
    print("Predictions:\n", results.pandas().xyxy[0])


if __name__ == "__main__":
    run_pytorch_yolov5_320()
