# MobileNetV3 Demo Script - TIMM (large)

import urllib

import pybuda
import requests
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def run_mobilenetv3_large_timm():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    # model = timm.create_model('mobilenetv3_large_100', pretrained=True)
    model = timm.create_model("hf_hub:timm/mobilenetv3_large_100.ra_in1k", pretrained=True)
    tt_model = pybuda.PyTorchModule("mobilenet_v3_large__hf_timm", model)

    # Image load and pre-processing into pixel_values
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    # STEP 3: Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get(timeout=0.5)

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0].value())[0]

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    run_mobilenetv3_large_timm()
