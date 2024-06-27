# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.tvm.cnn.pytorch.gscnn import get_model
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model




from test.tvm.cnn.pytorch.gscnn.wider_resnet import wider_resnet38_a2

def test_wider_resnet_torch(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    model = wider_resnet38_a2(classes=1000, dilation=True)
    submodel = torch.nn.Sequential(
        model.mod1,
        model.pool2,
    )

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH # Needs neg maxpool support tenstorrent/pybuda#188

    module = PyTorchModule("wider_resnet_torch", submodel)

    input_shape = (1, 3, 1024, 2048)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
