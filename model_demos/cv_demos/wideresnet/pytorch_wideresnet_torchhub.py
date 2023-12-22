# Wideresnet Demo Script

import os
import urllib

import pybuda
import torch
from PIL import Image
from torchvision import transforms


def run_wideresnet_torchhub_pytorch(variant="wide_resnet50_2"):
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Create PyBuda module from PyTorch model
    model = torch.hub.load("pytorch/vision:v0.10.0", variant, pretrained=True)

    model_name = f"pt_{variant}"

    tt_model = pybuda.PyTorchModule(model_name, model)

    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)

    # preprocessing
    input_image = Image.open(filename)
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

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get imagenet class mappings
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()
    print(result)


if __name__ == "__main__":
    run_wideresnet_torchhub_pytorch()
