# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Alexnet basic bring-up tests of tracing functionality
#
import time
import numpy as np
import pytest

import math
import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model
import pybuda

def test_tvm_alexnet(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if (test_kind == TestKind.TRAINING_RECOMPUTE):
        pytest.skip()  # tenstorrent/pybuda#215

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    if test_kind.is_training():
        compiler_cfg.compile_depth = (
            CompileDepth.GENERATE_INITIAL_GRAPH
        )  # Pooling backward is unimplemented

    pytorch_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True)
    module = PyTorchModule("pt_alexnet", pytorch_model)

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
