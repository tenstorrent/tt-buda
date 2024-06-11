# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import urllib
from test.utils import download_model
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pybuda._C.backend_api import BackendDevice


def generate_model_xception_imgcls_timm(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    if variant == "xception" and test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_policy = "CNN"
    else:
        compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_{variant}_timm", framework_model)

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return tt_model, [img_tensor]
