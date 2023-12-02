# U-Net Segmentation Models Pytorch Demo

import os
import urllib

import matplotlib.pyplot as plt
import pybuda
import requests
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, ConvertImageDtype, Normalize, PILToTensor, Resize


def get_imagenet_sample():

    # Get sample
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Preprocessing
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
    return img_tensor


def visualize(image):

    plt.imshow(image.squeeze(), cmap="RdBu")
    plt.colorbar()
    plt.close()


def download_model(download_func, *args, num_retries=3, timeout=120, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)

        except (requests.exceptions.HTTPError, urllib.error.HTTPError):
            logger.error("Failed to download the model after multiple retries.")
            assert False, "Failed to download the model after multiple retries."


def run_unet_qubvel_pytorch():

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1488"] = 3
    compiler_cfg.default_dram_parameters = False

    # Create PyBuda module from PyTorch model
    encoder_name = "resnet101"
    model = download_model(
        smp.Unet,
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    model.eval()
    tt_model = pybuda.PyTorchModule("unet_qubvel_pt", model)

    # Set model and processing parameters
    params = download_model(smp.encoders.get_preprocessing_params, encoder_name)
    std = torch.tensor(params["std"]).view(1, 3, 1, 1)
    mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

    # Preprocessing
    image = get_imagenet_sample()
    img_tensor = torch.tensor(image)
    img_tensor = (img_tensor - mean) / std

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]), _verify_cfg=pybuda.VerifyConfig())
    output = output_q.get()
    pr_mask = output[0].value()

    # Pass the raw output through sigmoid to get the probabilities
    pr_mask = pr_mask.sigmoid()
    pr_mask = (pr_mask > 0.5).float()
    pr_mask = pr_mask.detach().numpy()

    # Visualize output
    visualize(pr_mask)


if __name__ == "__main__":
    run_unet_qubvel_pytorch()
