# ResNet Demo Script

import os

import pybuda
import requests
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification


def run_resnet_pytorch(variant="microsoft/resnet-50"):

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = ResNetForImageClassification.from_pretrained(model_ckpt)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.enable_t_streaming = True
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"

    # Load data sample
    url = "https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/train/18/image/image.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = "tiger"

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_resnet50", model), inputs=[(pixel_values,)]
    )
    output = output_q.get()

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1).item()
    predicted_label = model.config.id2label[predicted_value]

    print(f"True Label: {label} | Predicted Label: {predicted_label}")


if __name__ == "__main__":
    run_resnet_pytorch()
