import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify import TestKind
from pybuda._C.backend_api import BackendDevice
import torch
import os
import pytest

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
    os.environ["PYBUDA_RIBBON2"] = "1"

    func = variants_func[variant]
    if test_device.arch == BackendDevice.Grayskull:
        if func.__name__ == "dla102x2":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    elif test_device.arch == BackendDevice.Wormhole_B0:
        if func.__name__ == "dla60x":
            compiler_cfg.place_on_new_epoch("concatenate_776.dc.concatenate.0")
        elif func.__name__ == "dla60x":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "20480"

    model_name = f"dla_{variant}_pytorch"
    input = torch.randn(1, 3, 384, 1280)

    pytorch_model = func(pretrained="imagenet")

    # Create pybuda.PyTorchModule using the loaded Pytorch model
    tt_model = pybuda.PyTorchModule(model_name, pytorch_model)

    verify_module(
        tt_model,
        input_shapes=[input.shape],
        inputs=[(input,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
