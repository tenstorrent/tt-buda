# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


def test_fcn_pytorch(test_kind, test_device):
    if test_device.is_silicon():
        pcc = 0.93
    else:
        pcc = 0.99

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    # Issue below is still valid, though it doesn't trigger when fracturing is turned on
    # tenstorrent/pybuda#310
    import pybuda
    pybuda.config.override_t_stream_shape(
        "conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (28, 1)
    )

    # tenstorrent/pybuda#392
    import os
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["PYBUDA_FORCE_RESIZE_DENSE_MM"] = "1"
    model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True, force_reload=True
    )
    module = PyTorchModule("fcn_resnet50", model)

    input_shape = (1, 3, 224, 224)
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
