# VGG Demo Script

import os
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms


def run_vgg_osmr_pytorch(variant="vgg11"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if variant in ["vgg11", "vgg13", "vgg16", "vgg19", "bn_vgg19", "bn_vgg19b"]:
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Create PyBuda module from PyTorch model
    model_ckpt = variant
    model_name = f"{model_ckpt}_osmr"
    model = ptcv_get_model(model_ckpt, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule(model_name, model)

    # Image preprocessing
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
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([input_batch]))
    output = output_q.get()[0].value()

    # The output has unnormalized scores. To get probabilities, you can run softmax on it.
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
    run_vgg_osmr_pytorch()
