# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# ConvNeXt basic bring-up tests of tracing functionality
#
import pytest

from transformers import ConvNextModel

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model
import pybuda


def test_convnext_tiny(test_kind, test_device):
    pytest.skip()
    # Always run with recompute in post-commit CI. Nightly tests both
    if test_kind == TestKind.TRAINING:
        pytest.skip()
        
    # import os
    # os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_AT"] = "PRE_PLACER"
    # os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "forward_only"
    # os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "backward_only"

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # tenstorrent/pybuda#365
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.retain_tvm_python_files = True

    framework_model = download_model(ConvNextModel.from_pretrained, "facebook/convnext-tiny-224", torchscript=True)
    module = PyTorchModule("pt_convnext_tiny", framework_model)

    input_shape = (1, 3, 64, 64)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.97,
        ),
    )


def test_convnext_embeddings(test_kind, test_device):
    pytest.skip() # Already testing with full model

    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = (
        CompileDepth.BUDA_GRAPH_PRE_PLACER
    )  # Unsupported HW ops

    framework_model = download_model(ConvNextModel.from_pretrained, "facebook/convnext-tiny-224", torchscript=True)
    framework_model = framework_model.embeddings
    module = PyTorchModule("convnext_embeddings", framework_model)

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


def test_convnext_layer(test_kind, test_device):
    pytest.skip() # Already testing with full model

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.compile_depth = (
        CompileDepth.BUDA_GRAPH_PRE_PLACER
    )  # Unsupported HW ops

    framework_model = download_model(ConvNextModel.from_pretrained, "facebook/convnext-tiny-224", torchscript=True)
    framework_model = framework_model.encoder.stages[0].layers[0]
    module = PyTorchModule("convnext_layer", framework_model)

    input_shape = (1, 96, 64, 64)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
