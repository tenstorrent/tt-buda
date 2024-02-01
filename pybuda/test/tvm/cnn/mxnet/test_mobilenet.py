# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
import pytest
import torch
from pybuda import (
    MXNetModule,
    BackendType,
    VerifyConfig,
)
from pybuda.config import CompileDepth

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


def test_tvm_mobilenet1_mxnet(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()


    model = download_model(get_model, "mobilenet1.0", pretrained=True)
    mod = MXNetModule(
        "mobilenet1_mxnet",
        model,
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
        )
    )

def test_tvm_mobilenet0_mxnet(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()


    model = download_model(get_model, "mobilenet0.25", pretrained=True)
    mod = MXNetModule(
        "mobilenet0_25_mxnet",
        model,
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
        )
    )
