import pybuda, os
import torch
from torchvision import transforms
import requests
from PIL import Image
import pytest
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda import VerifyConfig
import sys
from pybuda._C.backend_api import BackendDevice

sys.path.append("third_party/confidential_customer_models/generated/scripts/")
from model_ddrnet import DualResNet_23, DualResNet_39, BasicBlock


variants = ["ddrnet23s", "ddrnet23", "ddrnet39"]


@pytest.mark.parametrize("variant", variants)
def test_ddrnet_pytorch(variant, test_device):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    # STEP 2: Create PyBuda module from PyTorch model
    if variant == "ddrnet23s":

        model = DualResNet_23(
            block=BasicBlock, layers=[2, 2, 2, 2], planes=32, last_planes=1024
        )

    elif variant == "ddrnet23":

        model = DualResNet_23(
            block=BasicBlock, layers=[2, 2, 2, 2], planes=64, last_planes=2048
        )

    elif variant == "ddrnet39":

        model = DualResNet_39(
            block=BasicBlock, layers=[3, 4, 6, 3], planes=64, last_planes=2048
        )

    state_dict_path = (
        f"third_party/confidential_customer_models/generated/files/{variant}.pth"
    )

    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

    model.load_state_dict(state_dict, strict=False)

    model.eval()

    model_name = f"pt_{variant}"

    tt_model = pybuda.PyTorchModule(model_name, model)

    # STEP 3: Prepare input
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
    input_batch = input_tensor.unsqueeze(0)

    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=(
                0.98
                if test_device.arch == BackendDevice.Grayskull
                and variant != "ddrnet23s"
                else 0.99
            ),
        ),
    )
