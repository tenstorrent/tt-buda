# MobileNetV2 Demo Script - 96x96

import pybuda
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def run_mobilenetv2_hf(variant="google/mobilenet_v2_0.35_96"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Device specific configurations
    if variant == "google/mobilenet_v2_0.35_96" or "google/mobilenet_v2_0.75_160" or "google/mobilenet_v2_1.0_224":
        compiler_cfg.balancer_policy = "Ribbon"
        compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
        compiler_cfg.enable_t_streaming = True

    # Create PyBuda module from PyTorch model
    model_ckpt = variant
    preprocessor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModelForImageClassification.from_pretrained(model_ckpt)
    tt_model = pybuda.PyTorchModule("mobilenet_v2__hf", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([inputs.pixel_values]))
    output = output_q.get(timeout=0.5)

    # Data postprocessing
    predicted_class_idx = output[0].value().detach().float().numpy().argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


if __name__ == "__main__":
    run_mobilenetv2_hf()
