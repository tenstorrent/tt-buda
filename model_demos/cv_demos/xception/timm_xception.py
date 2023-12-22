# Xception

import os
import urllib

import pybuda
import timm
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def run_xception_timm(variant="xception"):
    """
    Variants = {
     'xception',
     'xception41',
     'xception65',
     'xception71'
    }
    """

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    available_devices = pybuda.detect_available_devices()

    if variant == "xception" and available_devices[0] == BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_policy = "CNN"
    else:
        compiler_cfg.balancer_policy = "Ribbon"

    model_name = variant
    model = timm.create_model(model_name, pretrained=True)

    # preprocessing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(f"{variant}_timm_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([tensor]))
    output = output_q.get()[0].value()

    # postprocessing
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get imagenet class mappings
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)

    # Get imagenet class mappings
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    run_xception_timm()
