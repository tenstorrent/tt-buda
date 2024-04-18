# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import numpy as np
import torch
import os
import skimage
import requests
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendDevice


def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    mean, std = 128, 128
    img = skimage.img_as_float(skimage.io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0, 2)
    return img


def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if aspect < 1:
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if aspect == 1:
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled


def crop_center(img, cropx, cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img


def prepare_input(img_uri):
    img = load_image(img_uri)
    img = rescale(img, 300, 300)
    img = crop_center(img, 300, 300)
    img = normalize(img)
    return img


def test_pytorch_ssd300_resnet50(test_device):

    # STEP 1 : Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.amp_level = 1

    if test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_RIBBON2"] = "1"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "90112"

    if test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.place_on_new_epoch("conv2d_766.dc.matmul.11")
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "45056"

    # STEP 2 : prepare model
    model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False
    )
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"

    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tt_model = pybuda.PyTorchModule("ssd300_resnet50", model)

    # STEP 3 : prepare input
    img = "http://images.cocodataset.org/val2017/000000397133.jpg"
    HWC = prepare_input(img)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    input_batch = torch.from_numpy(batch).float()

    # STEP 4 : Inference
    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.96 if test_device.arch == BackendDevice.Wormhole_B0 else 0.98,
        ),
    )
