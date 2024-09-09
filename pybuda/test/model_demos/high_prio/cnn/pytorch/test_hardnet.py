# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda, os
import torch
import pytest
import urllib
from PIL import Image
from torchvision import transforms
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendDevice

variants = ["hardnet68", "hardnet85", "hardnet68ds", "hardnet39ds"]


@pytest.mark.parametrize("variant", variants)
def test_hardnet_pytorch(test_device, variant):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if variant == "hardnet85" and test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    if variant == "hardnet68ds" and test_device.arch == BackendDevice.Grayskull:
        os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
        os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

    if test_device.arch == BackendDevice.Blackhole:
        if variant == "hardnet68ds":
            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        elif variant == "hardnet85":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "48000"

    # load only the model architecture without pre-trained weights.
    model = torch.hub.load("PingoLH/Pytorch-HarDNet", variant, pretrained=False)

    # load the weights downloaded from https://github.com/PingoLH/Pytorch-HarDNet
    checkpoint_path = (
        f"third_party/confidential_customer_models/internal/hardnet/files/weights/{variant}.pth"
    )

    # Load weights from the checkpoint file and maps tensors to CPU, ensuring compatibility even without a GPU.
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Inject weights into model
    model.load_state_dict(state_dict)
    model.eval()

    # STEP 2: Create PyBuda module from PyTorch model
    model_name = f"pt_{variant}"
    tt_model = pybuda.PyTorchModule(model_name, model)

    # STEP 3: Prepare input
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # Preprocessing
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
    input_batch = input_tensor.unsqueeze(0)

    pcc = (
        0.99
        if variant in ["hardnet68ds", "hardnet39ds"]
        and test_device.arch == BackendDevice.Wormhole_B0
        else 0.97
    )

    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )
