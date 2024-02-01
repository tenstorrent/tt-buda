# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
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


def test_deeplabv3_pytorch(test_kind, test_device):
    pytest.skip()  # Running full models on nightly

    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = (
        CompileDepth.BUDA_GRAPH_PRE_PLACER
    )  # Unsupported HW ops

    model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
    )
    module = PyTorchModule("deeplabv3_resnet50", model)

    input_shape = (1, 3, 224, 224)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
