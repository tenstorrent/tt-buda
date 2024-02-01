# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np

from pybuda import (
    TTDevice,
    Tensor,
    Parameter,
    PyBudaModule,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
    SGD,
)
from pybuda.utils import get_pybuda_parameters_from_state_dict
from pybuda._C.backend_api import BackendType

import pybuda.op as nn

from transformers.models.gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2MLP, GPT2Attention
from gpt2 import (
    get_default_gpt2_config,
    functional_gpt2_feedforward,
    functional_gpt2_mha,
    functional_gpt2_block,
    PyBudaGPT2MLP,
    PyBudaGPT2MHA,
    PyBudaGPT2Block,
    PyBudaGPT2LayerNorm,
)

verify_cfg=VerifyConfig(atol=1e-02, rtol=1e-02, run_golden=False) # don't run backend golden on each test yet

DEFAULT_SEQUENCE_LENGTH = 128


def split_qkv_weights_and_bias_aot(config, parameters, prefix=None):
    if prefix:
        qkv_weights_string = f"{prefix}.c_attn.weight"
        qkv_bias_string = f"{prefix}.c_attn.bias"
    else:
        qkv_weights_string = "c_attn.weight"
        qkv_bias_string = "c_attn.bias"

    qkv_suffix_string = ["query", "key", "value"]

    qkv_weights = parameters[qkv_weights_string]
    qkv_bias = parameters[qkv_bias_string]

    split_qkv_weights_tuple = qkv_weights.split(config.hidden_size, dim=-1)
    split_qkv_bias_tuple = qkv_bias.split(config.hidden_size, dim=-1)

    for suffix, tensor in zip(qkv_suffix_string, split_qkv_weights_tuple):
        parameters[f"{qkv_weights_string}.{suffix}"] = tensor
    for suffix, tensor in zip(qkv_suffix_string, split_qkv_bias_tuple):
        parameters[f"{qkv_bias_string}.{suffix}"] = tensor

    return parameters


def add_causal_mask_constants_to_parameters(config, parameters, prefix=None):
    # TODO(jchu):  parameterize seq_len
    if prefix:
        valid_causal_mask_key = f"{prefix}.valid_causal_mask"
        invalid_causal_mask_key = f"{prefix}.invalid_causal_mask"
    else:
        valid_causal_mask_key = "valid_causal_mask"
        invalid_causal_mask_key = "invalid_causal_mask"

    masked_bias = -1e4
    parameters[valid_causal_mask_key] = torch.tril(torch.ones(1, 1, 128, 128))
    parameters[valid_causal_mask_key].requires_grad = False
    parameters[invalid_causal_mask_key] = (
        torch.triu(torch.ones(1, 1, 128, 128), diagonal=1) * masked_bias
    )
    parameters[invalid_causal_mask_key].requires_grad = False


    return parameters


def test_functional_gpt2_mlp_vs_transformers():
    config = get_default_gpt2_config()
    mlp = GPT2MLP(config.hidden_size, config)

    activations = torch.rand(DEFAULT_SEQUENCE_LENGTH, config.hidden_size)

    golden = mlp(activations)
    calculated = functional_gpt2_feedforward(activations, mlp.state_dict())
    assert torch.allclose(golden, calculated)


def test_functional_gpt2_attn_vs_transformers():
    config = get_default_gpt2_config()
    gpt2_mha = GPT2Attention(config)

    parameters = split_qkv_weights_and_bias_aot(config, gpt2_mha.state_dict())
    parameters = add_causal_mask_constants_to_parameters(config, parameters)

    activations = torch.rand(1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    golden = gpt2_mha.forward(activations)
    calculated = functional_gpt2_mha(activations, config, parameters)
    assert torch.allclose(golden[0], calculated)


def test_functional_gpt2_attn_split_qkv():
    config = get_default_gpt2_config()
    gpt2_mha = GPT2Attention(config)

    parameters = split_qkv_weights_and_bias_aot(config, gpt2_mha.state_dict())
    parameters = add_causal_mask_constants_to_parameters(config, parameters)

    activations = torch.rand(1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    functional = functional_gpt2_mha(activations, config, parameters)
    functional_with_split = functional_gpt2_mha(
        activations, config, parameters, is_qkv_weight_split=True
    )
    assert torch.allclose(functional, functional_with_split)


def test_functional_gpt2_block_vs_transformers():
    config = get_default_gpt2_config()
    gpt2_block = GPT2Block(config)

    parameters = split_qkv_weights_and_bias_aot(config, gpt2_block.state_dict(), prefix="attn")
    parameters = add_causal_mask_constants_to_parameters(config, parameters, prefix="attn")

    activations = torch.rand(1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    golden = gpt2_block.forward(activations)
    calculated = functional_gpt2_block(activations, config, parameters, is_qkv_weight_split=True)
    assert torch.allclose(golden[0], calculated)


def test_pybuda_gpt2_feedforward():
    config = get_default_gpt2_config()
    mlp = GPT2MLP(config.hidden_size, config)
    pybuda_module = PyBudaGPT2MLP(
        "ff", get_pybuda_parameters_from_state_dict(mlp.state_dict()), config
    )

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(pybuda_module)

    activations = Tensor.create_from_torch(
        torch.rand(1, 1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    )

    # Adjust atol/rtol due to differences in gelu backwards implementation
    ret = pybuda_compile(
        tt0,
        "gpt2_ff",
        activations,
        compiler_cfg=CompilerConfig(
            enable_training=False,
            enable_recompute=False,
        ),
        verify_cfg=verify_cfg,
    )
    calculated = functional_gpt2_feedforward(activations.value(), mlp.state_dict())
    assert torch.allclose(ret.golden_outputs[0], calculated)


def test_pybuda_gpt2_mha():
    config = get_default_gpt2_config()
    gpt2_mha = GPT2Attention(config)

    parameters = split_qkv_weights_and_bias_aot(config, gpt2_mha.state_dict())
    parameters = add_causal_mask_constants_to_parameters(config, parameters)

    pybuda_module = PyBudaGPT2MHA(
        "mha", get_pybuda_parameters_from_state_dict(parameters), config
    )

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(pybuda_module)


    activations = Tensor.create_from_torch(
        torch.rand(1, 1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    )

    # Adjust atol/rtol due to differences in gelu backwards implementation
    ret = pybuda_compile(
        tt0,
        "gpt2_mha",
        activations,
        compiler_cfg=CompilerConfig(
            enable_training=False,
            enable_recompute=False,
        ),
        verify_cfg=verify_cfg,
    )
    pybuda_output = ret.golden_outputs[0].squeeze(dim=0) # squeeze to get back original tensor dims
    golden = functional_gpt2_mha(activations.value(), config, parameters)
    assert torch.allclose(pybuda_output, golden, atol=1e-02, rtol=1e-02)


def test_pybuda_gpt2_block():
    config = get_default_gpt2_config()
    gpt2_block = GPT2Block(config)

    parameters = split_qkv_weights_and_bias_aot(config, gpt2_block.state_dict(), prefix="attn")
    parameters = add_causal_mask_constants_to_parameters(config, parameters, prefix="attn")

    pybuda_module = PyBudaGPT2Block(
        "block", get_pybuda_parameters_from_state_dict(parameters), config
    )

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(pybuda_module)

    activations = Tensor.create_from_torch(
        torch.rand(1, 1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    )

    # Adjust atol/rtol due to differences in gelu backwards implementation
    ret = pybuda_compile(
        tt0,
        "gpt2_block",
        activations,
        compiler_cfg=CompilerConfig(
            enable_training=False,
            enable_recompute=False,
        ),
        verify_cfg=verify_cfg,
    )
    pybuda_output = ret.golden_outputs[0].squeeze(dim=0) # squeeze to get back original tensor dims
    golden = functional_gpt2_block(activations.value(), config, parameters)
    assert torch.allclose(pybuda_output, golden, atol=1e-02, rtol=1e-02)



def test_pybuda_gpt2_layernorm():
    config = get_default_gpt2_config()
    gpt2_block = GPT2Block(config)

    parameters = gpt2_block.state_dict()
    pybuda_parameters = get_pybuda_parameters_from_state_dict(parameters)

    pybuda_module = PyBudaGPT2LayerNorm(
        "ln", pybuda_parameters, config
    )

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(pybuda_module)

    activations = Tensor.create_from_torch(
        torch.rand(1, 1, DEFAULT_SEQUENCE_LENGTH, config.hidden_size)
    )

    # Adjust atol/rtol due to differences in gelu backwards implementation
    ret = pybuda_compile(
        tt0,
        "gpt2_layernorm",
        activations,
        compiler_cfg=CompilerConfig(
            enable_training=False,
            enable_recompute=False,
            balancer_policy="MinimizeGrid",
        ),
        verify_cfg=verify_cfg,
    )

    pybuda_output = ret.golden_outputs[0].squeeze(dim=0) # squeeze to get back original tensor dims

    golden = torch.nn.functional.layer_norm(
        activations.value(),
        (activations.value().shape[-1],),
        weight=parameters["ln_1.weight"],
        bias=parameters["ln_1.bias"],
    )

    assert torch.allclose(pybuda_output, golden, atol=1e-02, rtol=1e-02)
