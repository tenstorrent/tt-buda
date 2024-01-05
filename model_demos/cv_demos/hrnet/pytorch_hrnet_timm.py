# HRNet Demo Script

import os
import urllib

import pybuda
import requests
import timm
import torch
import torch.multiprocessing
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

torch.multiprocessing.set_sharing_strategy("file_system")


def run_hrnet_timm_pytorch(variant="hrnet_w18_small"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Device specific configurations
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    # Variant specific configurations
    if variant == "hrnet_w48":
        compiler_cfg.balancer_op_override("conv2d_343.dc.matmul.11", "grid_shape", (5, 3))
    elif variant == "hrnet_w64":
        compiler_cfg.place_on_new_epoch("add_1915")

    # Create PyBuda module from PyTorch model
    model = timm.create_model(variant, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_hrnet_timm_{variant}", model)

    # Preprocessing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    print(input_tensor.shape)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([input_tensor]))
    output = output_q.get()[0].value()

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
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
    run_hrnet_timm_pytorch()
