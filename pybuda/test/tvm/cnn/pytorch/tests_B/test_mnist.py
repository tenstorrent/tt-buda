# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda.compile import _get_global_compiler_config


def test_mnist_pytorch(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    class MNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=5 // 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=5 // 2)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(1024, 320)
            self.fc2 = nn.Linear(320, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 3, padding=3 // 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3, padding=3 // 2))
            x = x.view(-1, 1024)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, self.training)
            x = self.fc2(x)
            smx = F.softmax(x, dim=-1)

            return torch.log(smx)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    
    model = MNIST()
    module = PyTorchModule("mnist", model)

    input_shape = (1, 1, 32, 32)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
