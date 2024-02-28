import pybuda, os
import pytest
from torchvision import transforms
import requests
from PIL import Image
import onnx
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind

variants = ["ddrnet23s", "ddrnet23", "ddrnet39"]


@pytest.mark.parametrize("variant", variants)
def test_ddrnet(variant, test_device):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # STEP 2: # Create PyBuda module from onnx weights
    model_name = f"{variant}_onnx"

    load_path = (
        f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    )

    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule(model_name, model, load_path)

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
    img_tensor = input_tensor.unsqueeze(0)

    verify_module(
        tt_model,
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
        ),
    )
