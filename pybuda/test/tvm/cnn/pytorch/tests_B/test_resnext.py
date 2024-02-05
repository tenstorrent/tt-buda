# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
    BackendType,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


def test_resnext(test_kind, test_device):
    if test_device.devtype != BackendType.Silicon:
        pytest.skip()  # Takes too long in post commit

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    #import pybuda
    # tenstorrent/pybuda#310
    #pybuda.config.override_t_stream_shape("conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (28, 1))

    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext50_32x4d", pretrained=True)
    module = PyTorchModule("resnext50_32x4d", model)

    input_shape = (1, 3, 224, 224)
    pcc = 0.97 if test_device.devtype == BackendType.Silicon else 0.99
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
