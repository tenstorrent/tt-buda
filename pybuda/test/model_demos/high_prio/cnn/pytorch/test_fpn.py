import torch
import torch.nn as nn
import pybuda
import os
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict


class FPNWrapper(nn.Module):
    def __init__(
        self, in_channels_list, out_channels, extra_blocks=None, norm_layer=None
    ):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels, extra_blocks, norm_layer
        )

    def forward(self, feat0, feat1, feat2):
        x = OrderedDict()
        x["feat0"] = feat0
        x["feat1"] = feat1
        x["feat2"] = feat2
        return self.fpn(x)


def test_fpn_pytorch(test_device, test_kind):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Load FPN model
    model = FPNWrapper([10, 20, 30], 5)
    tt_model = pybuda.PyTorchModule("pytorch_fpn", model)

    feat0 = torch.rand(1, 10, 64, 64)
    feat1 = torch.rand(1, 20, 16, 16)
    feat2 = torch.rand(1, 30, 8, 8)

    verify_module(
        tt_model,
        input_shapes=[feat0.shape, feat1.shape, feat2.shape],
        inputs=[(feat0, feat1, feat2)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=test_kind,
        ),
    )
