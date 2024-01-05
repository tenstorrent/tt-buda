# Palm Detection Lite 1x1 demo

import os

import pybuda
import requests
import torch
from pybuda import TFLiteModule
from pybuda._C.backend_api import BackendDevice


def run_palm_detection_lite_1x1():

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
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"

    # Set PyBDUA environment variable
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Download model weights
    url = "https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite"
    tflite_path = "cv_demos/landmark/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(tflite_path, "wb") as f:
        f.write(response.content)

    # Load Palm Landmark model
    tt_model = TFLiteModule("tflite_palm_detection_lite", tflite_path)

    # Run inference on Tenstorrent device
    input_shape = (1, 192, 192, 3)
    input_tensor = torch.rand(input_shape)
    output_q = pybuda.run_inference(tt_model, inputs=([input_tensor]))
    output = output_q.get()
    print(output)

    # Remove weight file
    os.remove(tflite_path)


if __name__ == "__main__":
    run_palm_detection_lite_1x1()
