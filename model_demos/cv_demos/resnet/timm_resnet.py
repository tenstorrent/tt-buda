# ResNet50 - TIMM Demo Script

import urllib

import pybuda
import requests
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def run_resnet_timm():

    # Load ResNet50 feature extractor and model from TIMM
    model = timm.create_model("resnet50", pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Load data sample
    url = "https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/train/18/image/image.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    label = "tiger"

    # Data preprocessing
    pixel_values = transform(image).unsqueeze(0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(pybuda.PyTorchModule("timm_resnet50", model), inputs=[(pixel_values,)])
    output = output_q.get()

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0].value()[0], dim=0)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Get top-k prediction
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    predicted_label = categories[top1_catid]

    # Display output
    print(f"True Label: {label} | Predicted Label: {predicted_label} | Predicted Probability: {top1_prob.item():.2f}")


if __name__ == "__main__":
    run_resnet_timm()
