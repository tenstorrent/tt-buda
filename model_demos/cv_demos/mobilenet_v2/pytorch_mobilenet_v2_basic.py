# MobileNetV2 Demo Script - Basic

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from transformers import AutoImageProcessor


def run_mobilenetv2_basic():

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.balancer_policy = "Ribbon"
            compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
        elif available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True

    # STEP 2: Create PyBuda module from PyTorch model
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
    tt_model = pybuda.PyTorchModule("mobilenet_v2", model)

    # Image preprocessing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision,
    # to make a compatible postprocessing of the predicted class
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    img_tensor = preprocessor(images=image, return_tensors="pt").pixel_values

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get(timeout=0.5)

    # Data postprocessing
    predicted_class_idx = output[0].value().detach().float().numpy().argmax(-1).item()
    print("Predicted class:", predicted_class_idx)


if __name__ == "__main__":
    run_mobilenetv2_basic()
