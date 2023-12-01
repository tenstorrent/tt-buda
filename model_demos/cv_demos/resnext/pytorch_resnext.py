# ResneXt Demo Script

import os
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms


def get_image_tensor():
    # Image processing
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def run_resnext_pytorch(variant=("resnext14_32x4d", "osmr")):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Device specific configurations
    compiler_cfg.enable_t_streaming = True
    available_devices = pybuda.detect_available_devices()

    model_ckpt = variant[0]
    impl = variant[1]
    model_name = f"pt_{model_ckpt.replace('/', '_')}"
    if model_ckpt == "resnext14_32x4d":
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.balancer_policy = "Ribbon"
                os.environ["PYBUDA_RIBBON2"] = "1"
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            elif available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.balancer_policy = "CNN"
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{24*1024}"
    elif model_ckpt == "resnext26_32x4d":
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.balancer_policy = "Ribbon"
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
                os.environ["PYBUDA_RIBBON2"] = "1"
                os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
            elif available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.balancer_policy = "CNN"
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"
    elif model_ckpt == "resnext50_32x4d":
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_dram_parameters = False
                compiler_cfg.balancer_policy = "Ribbon"
                os.environ["PYBUDA_RIBBON2"] = "1"
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            elif available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.balancer_policy = "CNN"
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"
    elif model_ckpt == "resnext101_32x8d_wsl":
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.balancer_policy = "Ribbon"
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            elif available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.balancer_policy = "CNN"
                compiler_cfg.enable_auto_transposing_placement = True
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"
    elif model_ckpt == "resnext101_32x8d":
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.balancer_policy = "Ribbon"
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            elif available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.balancer_policy = "CNN"
                compiler_cfg.enable_auto_transposing_placement = True
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"
    elif model_ckpt == "resnext101_64x4d":
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_dram_parameters = False
                compiler_cfg.balancer_policy = "Ribbon"
                os.environ["PYBUDA_RIBBON2"] = "1"
                os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            elif available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.balancer_policy = "CNN"
                compiler_cfg.enable_auto_transposing_placement = True
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"

    # Create PyBuda module from PyTorch model
    if impl == "osmr":
        model = ptcv_get_model(model_ckpt, pretrained=True)
    else:
        model = torch.hub.load(impl, model_ckpt)

    model.eval()
    tt_model = pybuda.PyTorchModule(model_name, model)
    input_batch = get_image_tensor()

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
    run_resnext_pytorch()
