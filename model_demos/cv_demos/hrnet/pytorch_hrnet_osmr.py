# HRNet Demo Script

import os
import urllib

import pybuda
import requests
import torch
import torch.multiprocessing
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms

torch.multiprocessing.set_sharing_strategy("file_system")


def run_hrnet_osmr_pytorch(variant="hrnet_w18_small_v1"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Device specific configurations
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"

    # Variant specific configurations
    if variant in ["hrnetv2_w44", "hrnetv2_w48"]:
        compiler_cfg.balancer_op_override("conv2d_343.dc.matmul.11", "grid_shape", (5, 3))
    elif variant == "hrnetv2_w64":
        compiler_cfg.place_on_new_epoch("add_1915")

    # Create PyBuda module from PyTorch model
    model = ptcv_get_model(variant, pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_hrnet_osmr_{variant}", model)

    # Model load
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
    print(input_batch.shape)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([input_batch]))
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
    run_hrnet_osmr_pytorch()
