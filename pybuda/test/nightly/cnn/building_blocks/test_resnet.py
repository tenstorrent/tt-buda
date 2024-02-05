# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Script with pytests for ResNet and ResNeXt.

Read: https://arxiv.org/pdf/1512.03385.pdf, https://arxiv.org/pdf/1611.05431.pdf
"""

import pytest
import torch.nn as nn

from pybuda.config import _get_global_compiler_config
from pybuda import PyTorchModule, VerifyConfig
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice


class BasicResidualBlock(nn.Module):
    """BasicResidualBlock is the main building block of smaller ResNet archs."""

    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super().__init__()
        self.expansion = expansion

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.conv_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv_residual = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        # If residual connection has conv, pass the input through it.
        if self.conv_residual:
            x = self.conv_residual(x)

        y = y + x
        
        return self.relu(y)

class BottleneckResidualBlock(nn.Module):
    """BottleneckResidualBlock is the main building block of larger ResNets and all ResNeXt archs."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1, expansion=4):
        super().__init__()
        self.expansion = expansion

        self.conv1 = nn.Conv2d(in_channels, out_channels//self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//self.expansion)

        self.conv2 = nn.Conv2d(out_channels//self.expansion, out_channels//self.expansion, 
                               kernel_size=3, padding=1, groups=groups, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//self.expansion)

        self.conv3 = nn.Conv2d(out_channels//self.expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.conv_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv_residual = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        # If residual connection has conv, pass the input through it.
        if self.conv_residual:
            x = self.conv_residual(x)

        y = y + x

        return self.relu(y)

class ResnetInputBlock(nn.Sequential):
    """Both ResNet and ResNeXt archs share the same input block."""

    def __init__(self, in_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    @property
    def out_channels(self):
        return 64

class ResnetOutputBlock(nn.Sequential):
    """Both ResNet and ResNeXt archs share the same output block."""

    def __init__(self, block_expansion=1, num_classes=1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512*block_expansion, num_classes)
        )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("input_channels", [1, 3])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_resnet_input_block(input_size, input_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = ResnetInputBlock(input_channels)
    module = PyTorchModule("InputBlock", model)

    input_shape = (1, input_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_channels", [512, 2048])
@pytest.mark.parametrize("num_classes", [10, 100, 1000])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_resnet_output_block(input_channels, num_classes, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = ResnetOutputBlock(input_channels//512, num_classes)
    module = PyTorchModule("OutputBlock", model)

    input_shape = (1, input_channels, 7, 7)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, stride", 
    [(56, 64, 64, 1), (56, 64, 128, 2), (28, 128, 128, 1), (28, 128, 256, 2), 
     (14, 256, 256, 1), (14, 256, 512, 2), (7, 512, 512, 1)]
)
@pytest.mark.parametrize(
    "arch", 
    [BackendDevice.Grayskull, BackendDevice.Wormhole]
)
def test_resnet_basic_block(input_size, in_channels, out_channels, stride, arch):
    expected_to_fail = [
        (14, 256, 256, 1, BackendDevice.Grayskull),
        (7, 512, 512, 1, BackendDevice.Grayskull),
        (56, 64, 64, 1, BackendDevice.Wormhole),
        (28, 128, 128, 1, BackendDevice.Wormhole),
        (14, 256, 256, 1, BackendDevice.Wormhole),
        (7, 512, 512, 1, BackendDevice.Wormhole)
    ]

    if (input_size, in_channels, out_channels, stride, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = BasicResidualBlock(in_channels, out_channels, stride)
    module = PyTorchModule("ResNetBasicResidualBlock", model)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#369")
@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, stride, arch",
    [
        (14, 256, 256, 1, BackendDevice.Grayskull),
        (7, 512, 512, 1, BackendDevice.Grayskull),
        (56, 64, 64, 1, BackendDevice.Wormhole),
        (28, 128, 128, 1, BackendDevice.Wormhole),
        (14, 256, 256, 1, BackendDevice.Wormhole),
        (7, 512, 512, 1, BackendDevice.Wormhole)
    ]
)
def test_resnet_basic_block_xfail(input_size, in_channels, out_channels, stride, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = BasicResidualBlock(in_channels, out_channels, stride)
    module = PyTorchModule("ResNetBasicResidualBlock", model)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )   
    
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, stride",
    [(56, 64, 256, 1), (56, 256, 256, 1), (56, 256, 512, 2), (28, 512, 512, 1), 
     (28, 512, 1024, 2), (14, 1024, 1024, 1), (14, 1024, 2048, 2), (7, 2048, 2048, 1)]
)
@pytest.mark.parametrize(
    "arch", 
    [BackendDevice.Grayskull, BackendDevice.Wormhole]
)
def test_resnet_bottleneck_block(input_size, in_channels, out_channels, stride, arch):
    expected_to_fail = [
        (56, 256, 256, 1, BackendDevice.Grayskull),
        (28, 512, 512, 1, BackendDevice.Grayskull),
        (14, 1024, 1024, 1, BackendDevice.Grayskull),
        (7, 2048, 2048, 1, BackendDevice.Grayskull),
        (56, 256, 256, 1, BackendDevice.Wormhole),
        (28, 512, 512, 1, BackendDevice.Wormhole),
        (14, 1024, 1024, 1, BackendDevice.Wormhole),
        (7, 2048, 2048, 1, BackendDevice.Wormhole)
    ]

    if (input_size, in_channels, out_channels, stride, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = BottleneckResidualBlock(in_channels, out_channels, stride)
    module = PyTorchModule("ResNetBottleneckResidualBlock", model)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#369, "
                   "tenstorrent/pybuda#416, "
                   "tenstorrent/pybuda#417")
@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, stride, arch",
    [
        (56, 256, 256, 1, BackendDevice.Grayskull),
        (28, 512, 512, 1, BackendDevice.Grayskull),
        (14, 1024, 1024, 1, BackendDevice.Grayskull),
        (7, 2048, 2048, 1, BackendDevice.Grayskull),
        (56, 256, 256, 1, BackendDevice.Wormhole),
        (28, 512, 512, 1, BackendDevice.Wormhole),
        (14, 1024, 1024, 1, BackendDevice.Wormhole),
        (7, 2048, 2048, 1, BackendDevice.Wormhole)
    ]
)
def test_resnet_bottleneck_block_xfail(input_size, in_channels, out_channels, stride, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = BottleneckResidualBlock(in_channels, out_channels, stride)
    module = PyTorchModule("ResNetBottleneckResidualBlock", model)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, stride",
    [(56, 64, 256, 1), (56, 256, 256, 1), (56, 256, 512, 2), (28, 512, 512, 1), 
     (28, 512, 1024, 2), (14, 1024, 1024, 1), (14, 1024, 2048, 2), (7, 2048, 2048, 1)]
)
@pytest.mark.parametrize(
    "arch", 
    [BackendDevice.Grayskull, BackendDevice.Wormhole]
)
def test_resnext_bottleneck_block(input_size, in_channels, out_channels, stride, arch):
    expected_to_fail = [
        (56, 256, 256, 1, BackendDevice.Grayskull),
        (28, 512, 512, 1, BackendDevice.Grayskull),
        (14, 1024, 1024, 1, BackendDevice.Grayskull),
        (7, 2048, 2048, 1, BackendDevice.Grayskull),
        (56, 256, 256, 1, BackendDevice.Wormhole),
        (28, 512, 512, 1, BackendDevice.Wormhole),
        (14, 1024, 1024, 1, BackendDevice.Wormhole),
        (7, 2048, 2048, 1, BackendDevice.Wormhole)
    ]

    if (input_size, in_channels, out_channels, stride, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = BottleneckResidualBlock(in_channels, out_channels, stride, 32, 2)
    module = PyTorchModule("ResNeXtBottleneckResidualBlock", model)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#369, "
                   "tenstorrent/pybuda#416, "
                   "tenstorrent/pybuda#417")
@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, stride, arch",
    [
        (56, 256, 256, 1, BackendDevice.Grayskull),
        (28, 512, 512, 1, BackendDevice.Grayskull),
        (14, 1024, 1024, 1, BackendDevice.Grayskull),
        (7, 2048, 2048, 1, BackendDevice.Grayskull),
        (56, 256, 256, 1, BackendDevice.Wormhole),
        (28, 512, 512, 1, BackendDevice.Wormhole),
        (14, 1024, 1024, 1, BackendDevice.Wormhole),
        (7, 2048, 2048, 1, BackendDevice.Wormhole)
    ]
)
def test_resnext_bottleneck_block_xfail(input_size, in_channels, out_channels, stride, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = BottleneckResidualBlock(in_channels, out_channels, stride, 32, 2)
    module = PyTorchModule("ResNeXtBottleneckResidualBlock", model)

    input_shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.95
        ),
    )