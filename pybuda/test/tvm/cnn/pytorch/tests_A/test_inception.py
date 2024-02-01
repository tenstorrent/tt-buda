# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model



def test_inceptionv3_a_pytorch(test_kind, test_device):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "inception_v3", pretrained=True).Mixed_5b
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    module_name = str(model.__class__).split(".")[-1].split("'")[0]
    module = PyTorchModule(module_name, model)

    input_shape = (1, 192, 35, 35)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_inceptionv3_b_pytorch(test_kind, test_device):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "inception_v3", pretrained=True).Mixed_6a
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_conv_prestride = False  # tenstorrent/pybuda#925
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    module_name = str(model.__class__).split(".")[-1].split("'")[0]
    module = PyTorchModule(module_name, model)

    input_shape = (1, 288, 35, 35)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_inceptionv3_c_pytorch(test_kind, test_device):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "inception_v3", pretrained=True).Mixed_6b
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    module_name = str(model.__class__).split(".")[-1].split("'")[0]
    module = PyTorchModule(module_name, model)

    input_shape = (1, 768, 17, 17)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_inceptionv3_d_pytorch(test_kind, test_device):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "inception_v3", pretrained=True).Mixed_7a
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    module_name = str(model.__class__).split(".")[-1].split("'")[0]
    module = PyTorchModule(module_name, model)

    input_shape = (1, 768, 17, 17)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_inceptionv3_e_pytorch(test_kind, test_device):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "inception_v3", pretrained=True).Mixed_7b
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    module_name = str(model.__class__).split(".")[-1].split("'")[0]
    module = PyTorchModule(module_name, model)

    input_shape = (1, 1280, 8, 8)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    
def test_inceptionv3_full_pytorch(test_kind, test_device):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    module_name = str(model.Mixed_7b.__class__).split(".")[-1].split("'")[0]
    module = PyTorchModule(module_name, model)

    input_shape = (1, 3, 128, 128)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
