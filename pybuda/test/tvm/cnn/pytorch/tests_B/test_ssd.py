# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Single Shot MultiBox Detector (SSD) basic bring-up tests of tracing functionality
#
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from .SSD.ssd import SSD


def test_ssd(test_kind, test_device):
    pytest.skip()  # Testing full models only on nightly.

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    framework_model = SSD()
    input_shape = (1, 3, 300, 300)
    
    module = PyTorchModule(
        "ssd_full_model",
        framework_model
    )
    
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
