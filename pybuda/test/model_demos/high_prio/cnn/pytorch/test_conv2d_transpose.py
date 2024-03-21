import torch
import pybuda
import os

import pytest

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind

class Conv2d_transpose_model(torch.nn.Module):
    def __init__(self, in_channel,out_channel,kernel_size,stride,padding,groups):
        super().__init__()
        self.model = torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, 
                                                kernel_size=kernel_size, stride=stride, padding=padding, 
                                                output_padding=0, groups=groups, bias=False)

    def forward(self, input):
        return self.model(input)

@pytest.mark.skip(reason="Invalid place for this test")
def test_conv2d_transpose_0(test_device):

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Different in_channel and out_channel
    model = Conv2d_transpose_model(in_channel=256,out_channel=512,kernel_size=(4, 4),stride=(2, 2),padding=(1, 1),groups=1)
    model.eval()

    tt_model = pybuda.PyTorchModule("conv2d_transpose", model)
    input_shape = (1, 256, 12, 40)

    verify_module(
        tt_model,
        input_shapes=(input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )

@pytest.mark.skip(reason="Invalid place for this test")
def test_conv2d_transpose_1(test_device):

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Same in_channel and out_channel, but different groups
    model = Conv2d_transpose_model(in_channel=256,out_channel=256,kernel_size=(4, 4),stride=(2, 2),padding=(1, 1),groups=256)
    model.eval()

    tt_model = pybuda.PyTorchModule("conv2d_transpose", model)
    input_shape = (1, 256, 12, 40)

    verify_module(
        tt_model,
        input_shapes=(input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        )
    )
