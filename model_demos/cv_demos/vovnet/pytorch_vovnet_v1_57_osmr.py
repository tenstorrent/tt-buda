# VoVNet Model V1
"""
    VoVNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.
"""

import os
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms


def get_image():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(input_image)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def run_vovnet_v1_57_osmr_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Create PyBuda module from PyTorch model
    model = ptcv_get_model("vovnet57", pretrained=True)
    tt_model = pybuda.PyTorchModule("vovnet57_osmr_pt", model)

    # Run inference on Tenstorrent device
    img_tensor = get_image()
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()[0].value()

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()
    print(result)


if __name__ == "__main__":
    run_vovnet_v1_57_osmr_pytorch()
