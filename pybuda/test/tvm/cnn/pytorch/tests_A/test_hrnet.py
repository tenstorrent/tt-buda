# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# High-Resolution Network (HRNet) basic bring-up tests of tracing functionality
#
import pytest

import timm

from pybuda import (
    PyTorchModule,
    VerifyConfig,
    BackendType,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind


def test_hrnet_full_model(test_kind, test_device):
    if test_device.devtype != BackendType.Silicon:
        pytest.skip()  # Testing full model on nightly CI

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    framework_model = timm.create_model("hrnet_w18")
    module = PyTorchModule(
        "hrnet",
        framework_model,
    )

    input_shape = (1, 3, 64, 64)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_hrnet_basic_block(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = (
        CompileDepth.BUDA_GRAPH_PRE_PLACER
    )  # Unsupported HW ops

    framework_model = timm.create_model("hrnet_w18")
    framework_model = framework_model.stage2[0].branches[0][0]
    module = PyTorchModule(
        "hrnet_basic_block",
        framework_model,
    )

    input_shape = (1, 18, 9, 9)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_hrnet_fuse_layer(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = (
        CompileDepth.BUDA_GRAPH_PRE_PLACER
    )  # Unsupported HW ops

    framework_model = timm.create_model("hrnet_w18")
    framework_model = framework_model.stage2[0].fuse_layers[0][1]
    module = PyTorchModule(
        "hrnet_basic_block",
        framework_model,
    )

    input_shape = (1, 36, 9, 9)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
