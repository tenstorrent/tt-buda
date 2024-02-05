# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
from transformers import ViTModel, ViTConfig

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
import pybuda


def test_tvm_visual_transformer(test_kind, test_device):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()

    if test_kind.is_training():
        pytest.skip() 

    # Compiler configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    # Load model
    config = ViTConfig()
    config.num_attention_heads = 1
    config.num_hidden_layers = 1
    framework_model = ViTModel(config)
    pybuda_model = PyTorchModule("pt_visual_transformer", framework_model)
    
    # Sanity run
    input_shape = (1, 3, 224, 224)
    out = framework_model(torch.rand(input_shape))
    
    verify_module(
        pybuda_model,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_visual_transformer_encoder(test_kind, test_device):
    pytest.skip("Tested in full model test")
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip() 

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    config = ViTConfig()
    config.num_hidden_layers = 1
    model = ViTModel(config)
    module = PyTorchModule("VisualTransformerEncoder", model.encoder)

    input_shape = (1, 197, 768)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"layer.0.attention.attention.key.bias"},
            pcc=0.89
        ),
    )


def test_tvm_visual_transformer_pooler(test_kind, test_device):
    pytest.skip("Tested in full model test")
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    config = ViTConfig()
    config.num_hidden_layers = 1
    model = ViTModel(config)
    module = PyTorchModule("VisualTransformerPooler", model.pooler)

    input_shape = (1, 197, 768)
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
