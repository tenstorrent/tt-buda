# Pose Landmark Lite 1x1 demo

import os

import pybuda
import requests
import torch
from pybuda import TFLiteModule
from pybuda._C.backend_api import BackendDevice


def run_pose_landmark_lite_1x1():

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] != BackendDevice.Wormhole_B0:
            raise NotImplementedError("Model not supported on Grayskull")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_single_buffer_fallback = True

    # Set PyBDUA environment variable
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_SPLIT_RESIZE2D"] = "128"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_MAX_CONCAT_INPUTS"] = "6"

    # Download model weights
    url = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite"
    tflite_path = "cv_demos/landmark/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(tflite_path, "wb") as f:
        f.write(response.content)

    # Load Pose Landmark model
    tt_model = TFLiteModule("tflite_pose_landmark_light", tflite_path)

    # STEP 3: Run inference on Tenstorrent device
    input_shape = (1, 256, 256, 3)
    input_tensor = torch.rand(input_shape)
    output_q = pybuda.run_inference(tt_model, inputs=([input_tensor]))
    output = output_q.get()
    print(output)

    # Remove weight file
    os.remove(tflite_path)


if __name__ == "__main__":
    run_pose_landmark_lite_1x1()
