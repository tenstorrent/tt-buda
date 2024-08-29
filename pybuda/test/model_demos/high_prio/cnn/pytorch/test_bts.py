import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind

import torch
from torchvision import transforms
import numpy as np

from PIL import Image
import pytest
import os
import sys

sys.path = list(
    set(sys.path + ["third_party/confidential_customer_models/internal/bts/"])
)

from scripts.model import get_bts_model

# Clip produces invalid results in Silicon BackendType
# which leads to pcc drop in normalize op in BTS model
# So Disabling the verification in BBE for Silicon BackendType
# Issue link - https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2823

variants = ["densenet161_bts", "densenet121_bts"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_bts_pytorch(test_device, variant):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        if variant == "densenet161_bts":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "94564"
            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.balancer_op_override("multiply_196", "t_stream_shape", (1, 1))

        elif variant == "densenet121_bts":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "76356"
            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            compiler_cfg.enable_auto_fusing = False

    # Load sample image
    image_path = "third_party/confidential_customer_models/internal/bts/files/samples/rgb_00315.jpg"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = normalize(image)
    image = torch.unsqueeze(image, 0)

    # Get the model
    model = get_bts_model(variant)
    checkpoint = torch.load(
        "third_party/confidential_customer_models/internal/bts/files/weights/nyu/"
        + str(variant)
        + "/"
        + str(variant)
        + ".pt",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint)
    model.eval()

    class BtsModel_wrapper(torch.nn.Module):
        def __init__(self, model, focal):
            super().__init__()
            self.model = model
            self.focal = focal

        def forward(self, input_tensor):
            return self.model(input_tensor, self.focal)

    bts_model_wrapper = BtsModel_wrapper(model, focal=518.8579)
    bts_model_wrapper.eval()

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pt_" + str(variant), bts_model_wrapper)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(image.shape,)],
        inputs=[(image,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework=True,
            verify_tvm_compile=True,
            enabled=(
                False if test_device.devtype == pybuda.BackendType.Silicon else True
            ),
        ),
    )
