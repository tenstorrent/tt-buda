# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import onnx
import os
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
import requests
import pytest
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendDevice
import torchvision.transforms as transforms
from PIL import Image
import shutil


variants = [
    "dla34",
    "dla46_c",
    "dla46x_c",
    "dla60x_c",
    "dla60",
    "dla60x",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.parametrize("variant", variants)
def test_dla_onnx(test_device, variant):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    onnx_dir_path = "dla"
    onnx_model_path = f"dla/{variant}_Opset18.onnx"
    if not os.path.exists(onnx_model_path):
        if not os.path.exists("dla"):
            os.mkdir("dla")
        url = f"https://github.com/onnx/models/raw/main/Computer_Vision/{variant}_Opset18_timm/{variant}_Opset18.onnx?download="
        response = requests.get(url, stream=True)
        with open(onnx_model_path, "wb") as f:
            f.write(response.content)

    # Load DLA model
    model_name = f"dla_{variant}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    tt_model = pybuda.OnnxModule(model_name, onnx_model, onnx_model_path)

    pcc = 0.99
    if test_device.arch in [BackendDevice.Wormhole_B0, BackendDevice.Blackhole]:
        if variant == "dla34":
            pcc = 0.98
        elif variant == "dla169":
            pcc = 0.96
    elif test_device.arch == BackendDevice.Grayskull:
        if variant in ["dla46_c", "dla102x2", "dla169"]:
            pcc = 0.97
        if variant in ["dla60", "dla102x"]:
            pcc = 0.98
        if variant == "dla102x2":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor, )],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )

    # Cleanup model files
    os.remove(onnx_model_path)
    # os.rmdir(onnx_dir_path)
    shutil.rmtree(onnx_dir_path)
