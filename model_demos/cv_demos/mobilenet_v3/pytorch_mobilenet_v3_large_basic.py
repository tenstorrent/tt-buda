# MobileNetV3 Demo Script - Basic (large)

import os

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from transformers import AutoImageProcessor


def run_mobilenetv3_large_basic():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["PYBUDA_RIBBON2"] = "1"

    # Create PyBuda module from PyTorch model
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v3_large", pretrained=True)
    tt_model = pybuda.PyTorchModule("mobilenet_v3_large_pt", model)

    # Run inference on Tenstorrent device
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision, to make a compatible postprocessing of the predicted class
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    img_tensor = preprocessor(images=image, return_tensors="pt").pixel_values
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get(timeout=0.5)

    # Data postprocessing
    predicted_class_idx = output[0].value().detach().float().numpy().argmax(-1).item()
    print("Predicted class:", predicted_class_idx)


if __name__ == "__main__":
    run_mobilenetv3_large_basic()
