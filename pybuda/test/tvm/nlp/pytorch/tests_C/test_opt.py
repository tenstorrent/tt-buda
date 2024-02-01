# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from transformers import OPTModel, OPTConfig
# from transformers.models.opt.modeling_opt import XGLMAttention, ACT2FN
from pybuda import (
    PyTorchModule,
    BackendType,
    VerifyConfig,
)


from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_opt_decoder(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    if test_kind.is_training() and test_device.devtype == BackendType.Silicon:
        pytest.skip()

    configuration = OPTConfig()
    model = OPTModel(configuration)

    submodel = model.decoder.layers[0]
    mod = PyTorchModule("OPT_decoder_layer", submodel)

    relative_atol = 0.4 if test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.9 if test_device.devtype == BackendType.Silicon else 0.99
    input_shape = (1, 32, 768)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"self_attn.k_proj.bias"},
            relative_atol=relative_atol,
            pcc=pcc,
        )
    )


def test_opt_full(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    configuration = OPTConfig()
    configuration.return_dict = False
    model = OPTModel(configuration)

    submodel = model
    mod = PyTorchModule("OPT_full", submodel)

    relative_atol = 0.4 if test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.9 if test_device.devtype == BackendType.Silicon else 0.99
    input_shape = (1, 128)
    inputs = [torch.randint(0, configuration.vocab_size, input_shape)]
    verify_module(
        mod,
        (input_shape,),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"self_attn.k_proj.bias"},
            relative_atol=relative_atol,
            pcc=pcc,
        )
    )
