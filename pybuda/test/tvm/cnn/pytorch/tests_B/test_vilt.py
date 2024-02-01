# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

from transformers import ViltModel, ViltConfig

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind


def test_tvm_vision_language_transformer_encoder(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()
        
    #if test_kind.is_training():
    #    pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    config = ViltConfig()
    config.num_hidden_layers = 1
    model = ViltModel(config)
    module = PyTorchModule("VisLanguageTransformerEncoder", model.encoder)

    input_shape = (1, 197, 768)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"layer.0.attention.attention.key.bias"},
            pcc=0.9
        ),
    )


def test_tvm_vision_language_transformer_pooler(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    config = ViltConfig()
    config.num_hidden_layers = 1
    model = ViltModel(config)
    module = PyTorchModule("VisLanguageTransformerPooler", model.pooler)

    input_shape = (1, 197, 768)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.97
        ),
    )
