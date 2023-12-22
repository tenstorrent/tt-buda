# Wideresnet

import os
import urllib

import pybuda
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def run_wideresnet_timm_pytorch(variant="wide_resnet50_2"):
    """
    Variants = {
     'wide_resnet50_2',
     'wide_resnet101_2'
    }
    """

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    model_name = variant
    model = timm.create_model(model_name, pretrained=True)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    # Get imagenet class mappings
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)

    # Get imagenet class mappings
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(model_name + "_timm_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([tensor]))
    output = output_q.get()[0].value()

    # Postprocessing
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    run_wideresnet_timm_pytorch()
