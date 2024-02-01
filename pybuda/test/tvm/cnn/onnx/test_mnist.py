# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import onnx
import onnxruntime as ort
import pytest
import torch
from pybuda import (
    OnnxModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    TFGraphDefModule,
)
from pybuda.config import CompileDepth

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

import urllib
import os


def test_tvm_mnist_onnx(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()
        test_device.devtype = BackendType.NoBackend
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/mnist-12.onnx"

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx",
            save_path,
        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "mnist_onnx",
        onnx_model,
        save_path,
    )

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    input_shape = (1, 1, 28, 28)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    os.remove(save_path)