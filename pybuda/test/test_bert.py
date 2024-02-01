# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Bert-related tests for the new front-end
#

import math

import pytest
import torch

from pybuda import (
    TTDevice,
    BackendType,
    Tensor,
    Parameter,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
    SGD,
)

from .bert.modules import (
    PyBudaBertMHA,
    PyBudaFeedForward,
    PyBudaBertEncoder,
    PyBudaFFNorm,
    get_bert_parameters
)


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_mha(training, recompute):
    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    params = get_bert_parameters("mha", hidden_dim=128)

    config =  {
        "num_heads": 4,
        "encoder_index": 0,
    }
    mod = PyBudaBertMHA("mha", params, config)
    sgd_optimizer = SGD(learning_rate=0.5, parameters=mod.get_parameters())
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 1, 128, 128)
    encoder_input = Tensor.create_from_torch(torch.rand(shape))
    attention_mask = Tensor.create_from_torch(torch.rand(1, 1, 128, 1))

    for param in params.values():
        param.set_value( (torch.rand(*param.shape.get_pytorch_shape()) - 0.5).detach())
        param.value().requires_grad = param.requires_grad

    params["reciprocal_of_sqrt_of_head_size_0"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(4)))
    sgd_optimizer.set_optimizer_parameters()

    pybuda_compile(
        tt0,
        "bert_mha",
        encoder_input,
        attention_mask,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            enable_t_streaming=False,
        ),
        verify_cfg=VerifyConfig(pcc=0.99, rtol=1e-1, 
            waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"},
            run_net2pipe=not training),
    )


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_ff(training, recompute):
    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    params = get_bert_parameters("ff", hidden_dim=128)

    config =  {
        "num_heads": 4,
        "encoder_index": 0,
    }
    mod = PyBudaFeedForward("ff", params, config)
    sgd_optimizer = SGD(learning_rate=0.5, parameters=mod.get_parameters())
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 1, 128, 128)
    encoder_input = Tensor.create_from_torch(torch.rand(shape))

    for param in params.values():
        param.set_value((torch.rand(*param.shape.get_pytorch_shape()) - 0.5).detach())
        param.value().requires_grad = param.requires_grad
    sgd_optimizer.set_optimizer_parameters()

    # Adjust atol/rtol due to differences in gelu backwards implementation
    pybuda_compile(
        tt0,
        "bert_ff",
        encoder_input,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(atol=1e-02, rtol=1e-01, run_net2pipe=not training),
    )


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_ffnorm(training, recompute):
    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    params = get_bert_parameters("ffnorm", hidden_dim=128)

    config =  {
        "num_heads": 4,
        "encoder_index": 0,
    }
    mod = PyBudaFFNorm("ffnorm", params, config)
    sgd_optimizer = SGD(learning_rate=0.5, parameters=mod.get_parameters())
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 1, 128, 128)
    encoder_input = Tensor.create_from_torch(torch.rand(shape))

    for param in params.values():
        param.set_value( (torch.rand(*param.shape.get_pytorch_shape()) - 0.5).detach())
        param.value().requires_grad = param.requires_grad
    sgd_optimizer.set_optimizer_parameters()

    # Adjust atol/rtol due to differences in gelu backwards implementation
    pybuda_compile(
        tt0,
        "bert_ffnorm",
        encoder_input,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(atol=1e-02, rtol=1e-02, run_net2pipe=not training),
    )


@pytest.mark.skip(reason="Crashed on CI, can't reproduce locally")
@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_encoder(training, recompute):

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    params = get_bert_parameters("encoder", hidden_dim=128)

    config =  {
        "num_heads": 4,
        "encoder_index": 0,
    }
    mod = PyBudaBertEncoder("encoder", params, config)
    sgd_optimizer = SGD(learning_rate=0.5, parameters=mod.get_parameters())
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 1, 128, 128)
    encoder_input = Tensor.create_from_torch(torch.rand(shape))
    attention_mask = Tensor.create_from_torch(torch.rand(1, 1, 128, 1))

    for param in params.values():
        param.set_value( (torch.rand(*param.shape.get_pytorch_shape()) - 0.5).detach())
        param.value().requires_grad = param.requires_grad

    params["reciprocal_of_sqrt_of_head_size_0"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(4)))
    sgd_optimizer.set_optimizer_parameters()

    pybuda_compile(
        tt0,
        "bert_encoder",
        encoder_input,
        attention_mask,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(pcc=0.99, waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"}, run_net2pipe=not training),
    )
