import os

import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify import TestKind
from pybuda._C.backend_api import BackendDevice
import requests
import pytest
import torchvision.transforms as transforms
from PIL import Image

from test.model_demos.models.dla import (
    dla34,
    dla46_c,
    dla46x_c,
    dla60x_c,
    dla60,
    dla60x,
    dla102,
    dla102x,
    dla102x2,
    dla169,
)


variants_func = {
    "dla34": dla34,
    "dla46_c": dla46_c,
    "dla46x_c": dla46x_c,
    "dla60x_c": dla60x_c,
    "dla60": dla60,
    "dla60x": dla60x,
    "dla102": dla102,
    "dla102x": dla102x,
    "dla102x2": dla102x2,
    "dla169": dla169,
}
variants = list(variants_func.keys())


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dla_pytorch(variant, test_device):
    # PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    func = variants_func[variant]
    model_name = f"dla_{variant}_pytorch"

    pcc = 0.99
    if test_device.arch == BackendDevice.Wormhole_B0:
        if variant == ("dla60", "dla60x"):
            compiler_cfg.place_on_new_epoch("concatenate_776.dc.concatenate.0")
    elif test_device.arch == BackendDevice.Grayskull:
        if func.__name__ in ("dla102x2", "dla169"):
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        if func.__name__ == "dla46_c":
            pcc = 0.97

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

    pytorch_model = func(pretrained="imagenet")
    pytorch_model.eval()

    # Create pybuda.PyTorchModule using the loaded Pytorch model
    tt_model = pybuda.PyTorchModule(model_name, pytorch_model)

    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )
