# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# MobileNet v1 basic bring-up tests of tracing functionality
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
from pybuda import DataFormat
from test.utils import download_model


def test_mobilenet_v3_small(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    # tenstorrent/pybuda#392
    import os
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

    model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "mobilenet_v3_small", pretrained=True
    )
    module = PyTorchModule("mobilenet_v3_small", model,)

    if test_device.is_silicon():
        pcc = 0.8
    else:
        pcc = 0.99

    input_shape = (1, 3, 224, 224)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=pcc,
            fp32_fallback=DataFormat.Float16
        ),
    )


def test_mobilenet_v3_large(test_kind, test_device):
    pytest.skip()  # if small mobilenet passes than we assume that the larger also passes
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "mobilenet_v3_large", pretrained=True
    )
    module = PyTorchModule("mobilenet_v3_large", model,)

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
