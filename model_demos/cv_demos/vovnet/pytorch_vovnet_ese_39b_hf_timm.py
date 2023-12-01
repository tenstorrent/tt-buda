# VoVNet Model V2

import urllib

import pybuda
import requests
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Source
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vovnet.py
# 9 architecture variants are there, but loaded pre-trained weights for 2 variants only
# Vovnet V2 has a prefeix ese_* and Vovnet V1 without that prefix


def preprocess_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    return model, img_tensor


def run_vovnet_ese_39b_timm_pytorch():

    model_name = "ese_vovnet39b"
    model, img_tensor = preprocess_timm_model(model_name)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(model_name + "_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()[0].value()

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    run_vovnet_ese_39b_timm_pytorch()
