# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

from test.tvm.cnn.pytorch.fastdepth.models import MobileNetSkipAdd
from pybuda.config import CompileDepth, _get_global_compiler_config

def test_fastdepth_pytorch(test_kind, test_device):

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    model = MobileNetSkipAdd(pretrained=False)

    module = PyTorchModule("fastdepth_torch", model)

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

