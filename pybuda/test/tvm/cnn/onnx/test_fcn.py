# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import onnx
import pytest

from pybuda import (
    OnnxModule,
    BackendType,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import CompileDepth, _get_global_compiler_config


def test_tvm_fcn_onnx(test_kind, test_device):
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()
        test_device.devtype = BackendType.NoBackend
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/fcn-resnet50-12.onnx"

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx",
            save_path,
        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "fcn_resnet50_12_onnx",
        onnx_model,
        save_path,
    )

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    input_shape = (1, 3, 224, 224)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.6,
        ),
    )
    os.remove(save_path)
