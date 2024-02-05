# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# AutoEncoder basic bring-up tests of tracing functionality
#
import pytest

import torch
from torch import nn


from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

# SPDX-FileCopyrightText: Copyright (c) 2018 Udacity
#
# SPDX-License-Identifier: MIT
# https://github.com/udacity/deep-learning-v2-pytorch
class LinearAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_lin1 = nn.Linear(784, 128)
        self.encoder_lin2 = nn.Linear(128, 64)
        self.encoder_lin3 = nn.Linear(64, 12)
        self.encoder_lin4 = nn.Linear(12, 3)

        # Decoder
        self.decoder_lin1 = nn.Linear(3, 12)
        self.decoder_lin2 = nn.Linear(12, 64)
        self.decoder_lin3 = nn.Linear(64, 128)
        self.decoder_lin4 = nn.Linear(128, 784)

        self.act_fun = nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_lin1(x)
        act = self.act_fun(act)
        act = self.encoder_lin2(act)
        act = self.act_fun(act)
        act = self.encoder_lin3(act)
        act = self.act_fun(act)
        act = self.encoder_lin4(act)

        # Decode
        act = self.decoder_lin1(act)
        act = self.act_fun(act)
        act = self.decoder_lin2(act)
        act = self.act_fun(act)
        act = self.decoder_lin3(act)
        act = self.act_fun(act)
        act = self.decoder_lin4(act)

        return act


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_conv2d_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.encoder_conv2d_2 = nn.Conv2d(16, 4, 3, padding=1)
        self.encoder_max_pool2d = nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_conv2d_1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder_conv2d_2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        self.act_fun = nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_conv2d_1(x)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)
        act = self.encoder_conv2d_2(act)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)

        # Decode
        act = self.decoder_conv2d_1(act)
        act = self.act_fun(act)
        act = self.decoder_conv2d_2(act)

        return act


def test_linear_autoencoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN" 

    framework_model = LinearAE()
    module = PyTorchModule(
        "autoencoder",
        framework_model,
    )

    input_shape = (1, 784)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_conv_autoencoder(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    framework_model = ConvAE()
    module = PyTorchModule(
        "pt_conv_autoencoder",
        framework_model,
    )

    input_shape = (1, 3, 64, 64)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
