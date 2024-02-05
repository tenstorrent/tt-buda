# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import onnx
import pytest
import numpy as np
import onnxruntime

from pybuda import (
    OnnxModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import _get_global_compiler_config


def test_resnet_onnx(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # Skip training
    if test_kind.is_training():
        pytest.skip()

    # Download ONNX model
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/resnet50-v1-7.onnx"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-7.onnx?raw=true",
            save_path,
        )

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    pybuda_onnx_model = OnnxModule(
        "resnet50_v1_7_onnx",
        onnx_model,
        save_path,
    )

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    # Sanity run
    input_shape = (1, 3, 224, 224)

    # Compile and verify
    verify_module(
        pybuda_onnx_model,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    # Clean up
    os.remove(save_path)
