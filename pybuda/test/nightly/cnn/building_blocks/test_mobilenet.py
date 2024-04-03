# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Script with pytests for MobileNet v1 and v2.

Read: https://arxiv.org/pdf/1704.04861.pdf, https://arxiv.org/pdf/1801.04381.pdf
"""

import pytest
from torch import nn

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice


class DepthwiseSeparableConv(nn.Sequential):
    """Main building block of MobileNet v1."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        )

class ConvBatchnormRelu(nn.Module):
    """(Conv + BatchNorm + ReLU) block, used for InvertedResidual block for MobileNet v2."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, groups=1, use_activation=True
    ):
        super().__init__()
        padding = int((kernel_size - 1)/2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                              padding=padding, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

        if use_activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        
        if self.act:
            y = self.act(y)

        return y

class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNet v2."""

    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()

        self.use_residual = (stride == 1) and (in_channels == out_channels)
        expanded_channels = out_channels * expansion_factor

        self.expand_1x1 = ConvBatchnormRelu(
            in_channels, 
            expanded_channels, 
            kernel_size=1
        )

        self.conv_3x3 = ConvBatchnormRelu(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels
        )

        self.reduce_1x1 = ConvBatchnormRelu(
            expanded_channels,
            out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, x):
        y = self.expand_1x1(x)
        y = self.conv_3x3(y)
        y = self.reduce_1x1(y)

        return y + x if self.use_residual else y

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "image_size", 
    (224, 192, 160, 128), 
    ids=["image_size_224", "image_size_192", "image_size_160", "image_size_128"]
)
@pytest.mark.parametrize(
    "input_size_divider, in_channels_base, out_channels_base, stride",
    [(2, 32, 64, 1), (2, 64, 128, 2), (4, 128, 128, 1), (4, 128, 256, 2), (8, 256, 256, 1), 
     (8, 256, 512, 2), (16, 512, 512, 1), (16, 512, 1024, 2), (32, 1024, 1024, 2)]
)
@pytest.mark.parametrize(
    "width_multiplier", 
    (1, 0.75, 0.25), 
    ids=["width_multiplier_1", "width_multiplier_0_75", "width_multiplier_0_25"]
)
@pytest.mark.parametrize(
    "arch", 
    (BackendDevice.Grayskull, BackendDevice.Wormhole_B0), 
    ids=["Grayskull", "Wormhole_B0"]
)
def test_mobilenet_v1_depthwise_separable_conv(
    image_size, 
    input_size_divider, 
    in_channels_base, 
    out_channels_base, 
    stride,
    width_multiplier, 
    arch
):
    expected_to_fail = [
        (224, 2, 32, 64, 1, 1, BackendDevice.Grayskull),
        (192, 2, 32, 64, 1, 1, BackendDevice.Grayskull),
        (160, 2, 32, 64, 1, 1, BackendDevice.Grayskull),
        (224, 2, 32, 64, 1, 0.75, BackendDevice.Grayskull),
        (192, 2, 32, 64, 1, 0.75, BackendDevice.Grayskull),
        (160, 2, 32, 64, 1, 0.75, BackendDevice.Grayskull),
        (224, 2, 32, 64, 1, 0.25, BackendDevice.Grayskull),
        (192, 2, 32, 64, 1, 0.25, BackendDevice.Grayskull),
        (160, 2, 32, 64, 1, 0.25, BackendDevice.Grayskull),
        (224, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (192, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (160, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (128, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (224, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (192, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (160, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (128, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (224, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
        (192, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
        (160, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
        (128, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
    ]

    if (image_size, input_size_divider, in_channels_base, out_channels_base, 
        stride, width_multiplier, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.") 

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    in_channels = int(in_channels_base * width_multiplier)
    out_channels = int(out_channels_base * width_multiplier)
    input_size = image_size // input_size_divider

    depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels, stride)
    module = PyTorchModule(f"MobileNet_v1_depthwise_separable_conv", depthwise_separable_conv)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        [input_shape],
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
        )
    )
    
@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#416")
@pytest.mark.parametrize(
    "image_size, input_size_divider, in_channels_base, out_channels_base, "
    "stride, width_multiplier, arch",
    [
        (224, 2, 32, 64, 1, 1, BackendDevice.Grayskull),
        (192, 2, 32, 64, 1, 1, BackendDevice.Grayskull),
        (160, 2, 32, 64, 1, 1, BackendDevice.Grayskull),
        (224, 2, 32, 64, 1, 0.75, BackendDevice.Grayskull),
        (192, 2, 32, 64, 1, 0.75, BackendDevice.Grayskull),
        (160, 2, 32, 64, 1, 0.75, BackendDevice.Grayskull),
        (224, 2, 32, 64, 1, 0.25, BackendDevice.Grayskull),
        (192, 2, 32, 64, 1, 0.25, BackendDevice.Grayskull),
        (160, 2, 32, 64, 1, 0.25, BackendDevice.Grayskull),
        (224, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (192, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (160, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (128, 2, 32, 64, 1, 1, BackendDevice.Wormhole_B0),
        (224, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (192, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (160, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (128, 2, 32, 64, 1, 0.75, BackendDevice.Wormhole_B0),
        (224, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
        (192, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
        (160, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
        (128, 2, 32, 64, 1, 0.25, BackendDevice.Wormhole_B0),
    ]
)
def test_mobilenet_v1_depthwise_separable_conv_xfail(
    image_size, 
    input_size_divider, 
    in_channels_base, 
    out_channels_base, 
    stride,
    width_multiplier, 
    arch
):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    in_channels = int(in_channels_base * width_multiplier)
    out_channels = int(out_channels_base * width_multiplier)
    input_size = image_size // input_size_divider

    depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels, stride)
    module = PyTorchModule(f"MobileNet_v1_depthwise_separable_conv", depthwise_separable_conv)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        [input_shape],
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
        )
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "image_size", 
    (224, 192, 160, 128), 
    ids=["image_size_224", "image_size_192", "image_size_160", "image_size_128"]
)
@pytest.mark.parametrize(
    "input_size_divider, in_channels_base, out_channels_base, stride",
    [(2, 32, 16, 1), (2, 16, 24, 2), (4, 24, 32, 2), (8, 32, 64, 2), 
     (16, 64, 96, 1), (16, 96, 160, 2), (32, 160, 320, 1)]
)
@pytest.mark.parametrize(
    "width_multiplier", 
    (1, 0.75, 0.25), 
    ids=["width_multiplier_1", "width_multiplier_0_75", "width_multiplier_0_25"]
)
@pytest.mark.parametrize(
    "expansion_factor", 
    (5, 6, 7), 
    ids=["expansion_factor_5", "expansion_factor_6", "expansion_factor_7"]
)
@pytest.mark.parametrize(
    "arch", 
    (BackendDevice.Grayskull, BackendDevice.Wormhole_B0), 
    ids=["Grayskull", "Wormhole_B0"]
)
def test_mobilenet_v2_inverted_residual(
    image_size, 
    input_size_divider, 
    in_channels_base, 
    out_channels_base, 
    stride, 
    width_multiplier, 
    expansion_factor, 
    arch
):
    expected_to_fail = [
        (192, 16, 64, 96, 1, 0.25, 6, BackendDevice.Grayskull),
        (224, 32, 160, 320, 1, 0.75, 7, BackendDevice.Grayskull),
        (192, 32, 160, 320, 1, 0.75, 7, BackendDevice.Grayskull),
        (192, 16, 64, 96, 1, 0.25, 6, BackendDevice.Wormhole_B0),
        (224, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0),
        (192, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0),
        (160, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0),
        (128, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0)
    ]

    if (image_size, input_size_divider, in_channels_base, out_channels_base, 
        stride, width_multiplier, expansion_factor, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.") 
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    in_channels = int(in_channels_base * width_multiplier)
    out_channels = int(out_channels_base * width_multiplier)
    input_size = image_size // input_size_divider

    inverted_residual = InvertedResidual(in_channels, out_channels, stride, expansion_factor)
    module = PyTorchModule(f"MobileNet_v2_inverted_residual", inverted_residual)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        [input_shape],
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
        )
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#417")
@pytest.mark.parametrize(
    "image_size, input_size_divider, in_channels_base, out_channels_base, "
    "stride, width_multiplier, expansion_factor, arch",
    [
        (192, 16, 64, 96, 1, 0.25, 6, BackendDevice.Grayskull),
        (224, 32, 160, 320, 1, 0.75, 7, BackendDevice.Grayskull),
        (192, 32, 160, 320, 1, 0.75, 7, BackendDevice.Grayskull),
        (192, 16, 64, 96, 1, 0.25, 6, BackendDevice.Wormhole_B0),
        (224, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0),
        (192, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0),
        (160, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0),
        (128, 32, 160, 320, 1, 0.75, 7, BackendDevice.Wormhole_B0)
    ]
)
def test_mobilenet_v2_inverted_residual_xfail(
    image_size, 
    input_size_divider, 
    in_channels_base, 
    out_channels_base, 
    stride, 
    width_multiplier, 
    expansion_factor, 
    arch
):  
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    in_channels = int(in_channels_base * width_multiplier)
    out_channels = int(out_channels_base * width_multiplier)
    input_size = image_size // input_size_divider

    inverted_residual = InvertedResidual(in_channels, out_channels, stride, expansion_factor)
    module = PyTorchModule(f"MobileNet_v2_inverted_residual", inverted_residual)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        [input_shape],
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
        )
    )