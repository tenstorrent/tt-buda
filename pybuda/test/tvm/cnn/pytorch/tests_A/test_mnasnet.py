# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from torchvision.models.mnasnet import MNASNet

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind


def test_mnasnet(test_kind, test_device):
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    
    if test_kind.is_training():
        pytest.skip()
    else:
        compiler_cfg.compile_depth = (
            CompileDepth.BUDA_GRAPH_PRE_PLACER
        )  # Unsupported HW ops

    framework_model = MNASNet(1.0)
    module = PyTorchModule(
        "mnasnet",
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
