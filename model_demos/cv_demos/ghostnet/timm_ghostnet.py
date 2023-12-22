# Ghostnet

import os
import urllib
from urllib.request import urlopen

import pybuda
import timm
import torch
from PIL import Image


def run_ghostnet_timm():
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    model = timm.create_model("ghostnet_100", pretrained=True)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("ghostnet_100_timm_pt", model)

    data_config = timm.data.resolve_data_config({}, model=model)
    transforms = timm.data.create_transform(**data_config)

    img = Image.open(urlopen("https://github.com/pytorch/hub/raw/master/images/dog.jpg"))
    img_tensor = transforms(img).unsqueeze(0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()[0].value()

    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

    # Get imagenet class mappings
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    for i in range(top5_probabilities.size(1)):
        class_idx = top5_class_indices[0, i].item()
        class_prob = top5_probabilities[0, i].item()
        class_label = categories[class_idx]

        print(f"{class_label} : {class_prob}")


if __name__ == "__main__":
    run_ghostnet_timm()
