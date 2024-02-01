# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
    BackendType,
)
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

import sys
import os
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.tvm.cnn.pytorch.videopose.model import TemporalModel

def test_videopose_pytorch(test_kind, test_device):

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    #if test_kind.is_training():
    #    pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    #compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    model = TemporalModel(
        17,
        2,
        17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False,
    )


    module = PyTorchModule("videopose_torch", model)

    pcc = 0.98 if test_device.devtype == BackendType.Silicon and test_kind.is_training() else 0.99
    input_shape = (1, 1024, 17, 2)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=pcc,
        ),
    )
