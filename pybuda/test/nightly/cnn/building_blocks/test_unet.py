# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Script with pytests for ViT.

Read: https://arxiv.org/pdf/1505.04597.pdf
"""

import pytest
import torch
import torch.nn as nn

from pybuda import PyTorchModule, VerifyConfig
from pybuda.module import PyTorchModule
from pybuda.config import _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendType, BackendDevice

class DoubleConvBatchnormRelu(nn.Module):
    """2x(Conv + BatchNorm + ReLU) block.
    
    NOTE Has many fails.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class DoubleConvRelu(nn.Module):
    """2x(Conv + ReLU) block.
    
    NOTE Implemented as a first workaround for many problems 
    DoubleConvBatchnormRelu has.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class DoubleConvBatchnormReluMaxpool(nn.Module):
    """2x(Conv + BatchNorm + ReLU) + MaxPool block.
    
    NOTE Implemented as a second workaround for many problems 
    DoubleConvBatchnormRelu has.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class Maxpool(nn.Module):
    """MaxPool block."""

    def __init__(self):
        super().__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)

class Upconv(nn.Module):
    """Upconv block."""

    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.layer(x)

class UpconvDoubleConvRelu(nn.Module):
    """Upconv + 2x(Conv + ReLU) block.
    
    NOTE Implemented as a workaround for Upconv while it was failing, but left
    here since it hit new problems.
    """

    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(input_channels//2, input_channels//4, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//4, input_channels//4, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class Concat(nn.Module):
    """Copy and crop / concat block."""

    def __init__(self):
        super().__init__()

    def forward(self, encoder_activations, upconv_activations):
        return self.__concat(upconv_activations, encoder_activations, -3)

    def __concat(self, x, y, dim):
        return torch.cat([x, y], dim=dim)

class UnityConv(nn.Module):
    """1x1 Conv block."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.layer(x)

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("in_channels, out_channels", 
                         [(3, 32), (32, 64), (64, 128), (128, 256), (256, 512)])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_double_conv_batchnorm_relu(input_size, in_channels, out_channels, arch):
    expected_to_fail = [
        (128, 32, 64, BackendDevice.Grayskull),
        (256, 32, 64, BackendDevice.Grayskull),
        (512, 32, 64, BackendDevice.Grayskull),
        (128, 64, 128, BackendDevice.Grayskull),
        (256, 64, 128, BackendDevice.Grayskull),
        (512, 64, 128, BackendDevice.Grayskull),
        (128, 128, 256, BackendDevice.Grayskull),
        (256, 128, 256, BackendDevice.Grayskull),
        (512, 128, 256, BackendDevice.Grayskull),
        (128, 256, 512, BackendDevice.Grayskull),
        (256, 256, 512, BackendDevice.Grayskull),
        (512, 256, 512, BackendDevice.Grayskull),
        (128, 32, 64, BackendDevice.Wormhole),
        (256, 32, 64, BackendDevice.Wormhole),
        (512, 32, 64, BackendDevice.Wormhole),
        (128, 64, 128, BackendDevice.Wormhole),
        (256, 64, 128, BackendDevice.Wormhole),
        (512, 64, 128, BackendDevice.Wormhole),
        (128, 128, 256, BackendDevice.Wormhole),
        (256, 128, 256, BackendDevice.Wormhole),
        (512, 128, 256, BackendDevice.Wormhole),
        (128, 256, 512, BackendDevice.Wormhole),
        (256, 256, 512, BackendDevice.Wormhole),
        (512, 256, 512, BackendDevice.Wormhole)
    ]

    if (input_size, in_channels, out_channels, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = DoubleConvBatchnormRelu(in_channels, out_channels)
    module = PyTorchModule("DoubleConvBatchnormRelu", model)

    shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        )
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#369")
@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, arch",
    [
        (128, 32, 64, BackendDevice.Grayskull),
        (256, 32, 64, BackendDevice.Grayskull),
        (512, 32, 64, BackendDevice.Grayskull),
        (128, 64, 128, BackendDevice.Grayskull),
        (256, 64, 128, BackendDevice.Grayskull),
        (512, 64, 128, BackendDevice.Grayskull),
        (128, 128, 256, BackendDevice.Grayskull),
        (256, 128, 256, BackendDevice.Grayskull),
        (512, 128, 256, BackendDevice.Grayskull),
        (128, 256, 512, BackendDevice.Grayskull),
        (256, 256, 512, BackendDevice.Grayskull),
        (512, 256, 512, BackendDevice.Grayskull),
        (128, 32, 64, BackendDevice.Wormhole),
        (256, 32, 64, BackendDevice.Wormhole),
        (512, 32, 64, BackendDevice.Wormhole),
        (128, 64, 128, BackendDevice.Wormhole),
        (256, 64, 128, BackendDevice.Wormhole),
        (512, 64, 128, BackendDevice.Wormhole),
        (128, 128, 256, BackendDevice.Wormhole),
        (256, 128, 256, BackendDevice.Wormhole),
        (512, 128, 256, BackendDevice.Wormhole),
        (128, 256, 512, BackendDevice.Wormhole),
        (256, 256, 512, BackendDevice.Wormhole),
        (512, 256, 512, BackendDevice.Wormhole)
    ]
)
def test_unet_double_conv_batchnorm_relu_xfail(input_size, in_channels, out_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = DoubleConvBatchnormRelu(in_channels, out_channels)
    module = PyTorchModule("DoubleConvBatchnormRelu", model)

    shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        )
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("in_channels, out_channels", 
                         [(3, 32), (32, 64), (64, 128), (128, 256), (256, 512)])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_double_conv_relu(input_size, in_channels, out_channels, arch):
    expected_to_fail = [(512, 256, 512, BackendDevice.Wormhole)]

    if (input_size, in_channels, out_channels, arch) in expected_to_fail:
        pytest.skip(msg="This combination is expected to fail, moved to _xfail version of the function.")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = DoubleConvRelu(in_channels, out_channels)
    module = PyTorchModule("DoubleConvRelu", model)

    shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        )
    )

@pytest.mark.xfail(reason="XFAIL due to: "
                   "tenstorrent/pybuda#422")
@pytest.mark.parametrize(
    "input_size, in_channels, out_channels, arch",
    [(512, 256, 512, BackendDevice.Wormhole)]
)
def test_unet_double_conv_relu_xfail(input_size, in_channels, out_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = DoubleConvRelu(in_channels, out_channels)
    module = PyTorchModule("DoubleConvRelu", model)

    shape = (1, in_channels, input_size, input_size)

    verify_module(
        module,
        (shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        )
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("in_channels, out_channels", 
                         [(3, 32), (32, 64), (64, 128), (128, 256), (256, 512)])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_double_conv_batchnorm_relu_maxpool(input_size, in_channels, out_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = DoubleConvBatchnormReluMaxpool(in_channels, out_channels)
    module = PyTorchModule("DoubleConvBatchnormReluMaxpool", model)

    shape = (1, in_channels, input_size, input_size)

    try:
        verify_module(
            module,
            (shape,),
            verify_cfg=VerifyConfig(
                arch=arch,
                devtype=BackendType.Golden,
                test_kind=TestKind.INFERENCE,
                pcc=0.96,
            )
        )
    except Exception as e:
        print(f"Failed arguments: ({input_size}, {in_channels}, {out_channels}, {arch})")
        raise e

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("input_channels", [1, 3, 32, 64, 128, 256])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_maxpool(input_size, input_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = Maxpool()
    module = PyTorchModule("Maxpool", model)

    shape = (1, input_channels, input_size, input_size)

    try:
        verify_module(
            module,
            (shape,),
            verify_cfg=VerifyConfig(
                arch=arch,
                devtype=BackendType.Golden,
                test_kind=TestKind.INFERENCE,
                pcc=0.96,
            ),
        )
    except Exception as e:
        print(f"Failed arguments: ({input_size}, {input_channels}, {arch})")
        raise e

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("input_channels", [256, 128, 64, 32])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_upconv(input_size, input_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = Upconv(input_channels)
    module = PyTorchModule("Upconv", model)

    shape = (1, input_channels, input_size, input_size)

    try:
        verify_module(
            module,
            (shape,),
            verify_cfg=VerifyConfig(
                arch=arch,
                devtype=BackendType.Golden,
                test_kind=TestKind.INFERENCE,
                pcc=0.96,
            ),
        )
    except Exception as e:
        print(f"Failed arguments: ({input_size}, {input_channels}, {arch})")
        raise e

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("input_channels", [256, 128, 64, 32])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_upconv_double_conv_relu(input_size, input_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = UpconvDoubleConvRelu(input_channels)
    module = PyTorchModule("UpconvDoubleConvRelu", model)

    shape = (1, input_channels, input_size, input_size)

    try:
        verify_module(
            module,
            (shape,),
            verify_cfg=VerifyConfig(
                arch=arch,
                devtype=BackendType.Golden,
                test_kind=TestKind.INFERENCE,
                pcc=0.96,
            ),
        )
    except Exception as e:
        print(f"Failed arguments: ({input_size}, {input_channels}, {arch})")
        raise e

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("encoder_activations_channels", [3, 32, 64])
@pytest.mark.parametrize("upconv_activations_channels", [3, 32, 64])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_concat(input_size, encoder_activations_channels, upconv_activations_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = Concat()
    module = PyTorchModule("Concat", model)

    encoder_activations_shape = (1, encoder_activations_channels, input_size, input_size)
    upconv_activations_shape = (1, upconv_activations_channels, input_size, input_size)

    verify_module(
        module,
        (encoder_activations_shape, upconv_activations_shape),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        ),
    )

# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_size", [128, 256, 512])
@pytest.mark.parametrize("input_channels", [128, 64, 32])
@pytest.mark.parametrize("output_channels", [3, 2, 1])
@pytest.mark.parametrize("arch", [BackendDevice.Grayskull, BackendDevice.Wormhole])
def test_unet_unityconv(input_size, input_channels, output_channels, arch):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = UnityConv(input_channels, output_channels)
    module = PyTorchModule("UnityConv", model)

    shape = (1, input_channels, input_size, input_size)

    verify_module(
        module,
        (shape,),
        verify_cfg=VerifyConfig(
            arch=arch,
            devtype=BackendType.Golden,
            test_kind=TestKind.INFERENCE,
            pcc=0.96,
        ),
    )