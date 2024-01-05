# DenseNet Demo Script

import os
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from torchvision.transforms import CenterCrop, Compose, ConvertImageDtype, Normalize, PILToTensor, Resize

torch.multiprocessing.set_sharing_strategy("file_system")


def get_input_img():

    # Get image
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Preprocessing
    img_tensor = transform(img).unsqueeze(0)
    print(img_tensor.shape)
    return img_tensor


def run_densenet_pytorch(variant="densenet121"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    if variant == "densenet121":
        compiler_cfg.enable_t_streaming = True
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_dram_parameters = False
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    elif variant == "densenet161":
        compiler_cfg.balancer_policy = "CNN"
        compiler_cfg.enable_t_streaming = True
        compiler_cfg.place_on_new_epoch("concatenate_131.dc.sparse_matmul.7.lc2")
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
        os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                # compiler_cfg.default_dram_parameters = True
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            elif available_devices[0] == BackendDevice.Grayskull:
                os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    elif variant == "densenet169":
        compiler_cfg.balancer_policy = "CNN"
        compiler_cfg.enable_t_streaming = True
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                # compiler_cfg.default_dram_parameters = False
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            else:
                os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"

    elif variant == "densenet201":
        compiler_cfg.balancer_policy = "CNN"
        compiler_cfg.enable_t_streaming = True
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            else:
                os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"

    # Create PyBuda module from PyTorch model
    model_ckpt = variant
    model = torch.hub.load("pytorch/vision:v0.10.0", model_ckpt, pretrained=True)
    tt_model = pybuda.PyTorchModule("densnet121_pt", model)

    # Run inference on Tenstorrent device
    img_tensor = get_input_img()
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()[0].value()

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0])
    print("PROB:", probabilities.shape)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    run_densenet_pytorch()
