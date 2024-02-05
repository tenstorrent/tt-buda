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

def test_efficientnet_layer(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_cfg.balancer_policy = "CNN"

    model = download_model(torch.hub.load, 
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=True,
    )
    module = PyTorchModule("efficientnet_b0_layer_torch", model.layers[0])

    input_shape = (1, 32, 112, 112)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_efficientnet_stem(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    #if test_kind.is_training():
    #    pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = download_model(torch.hub.load, 
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=True,
    )
    module = PyTorchModule("efficientnet_b0_stem_torch", model.stem)

    pcc = 0.98 if test_device.devtype == BackendType.Silicon and test_kind.is_training() else 0.99
    input_shape = (1, 3, 64, 64)
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

def test_efficientnet_b0(test_kind, test_device):
    pytest.skip()
    #if test_kind.is_training():
    #    pytest.skip()  # Backward is currently unsupported

    import timm
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = timm.create_model('efficientnet_b0', pretrained=True)
    module = PyTorchModule("efficientnet_b0", model)

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

def test_efficientnet_b4(test_kind, test_device):
    pytest.skip()
    #if test_kind.is_training():
    #    pytest.skip()  # Backward is currently unsupported

    import timm
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = timm.create_model('efficientnet_b4', pretrained=True)
    module = PyTorchModule("efficientnet_b0", model)

    input_shape = (1, 3, 320, 320)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
