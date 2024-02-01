# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np

from pybuda import (
    PyBudaModule,
)
import pybuda.op as nn

from transformers.models.gpt2 import GPT2Config


def get_default_gpt2_config():
    config = GPT2Config()
    config.activation_function = (
        "gelu"  # TODO: by default, config uses "gelu_new" which is updated gelu impl.
    )

    config.attn_pdrop = 0  # ignore dropout; for training
    config.resid_pdrop = 0  # ignore dropout; for training

    config.scale_attn_weights = False
    return config

# SPDX-FileCopyrightText: Copyright (c) 2018- The Hugging Face team.
#
# SPDX-License-Identifier: Apache-2.0
# https://github.com/huggingface/transformers
def functional_gpt2_feedforward(activations, parameters, prefix=None):
    """
    Note: The Transformers lib calls this a MLP module and implements this via conv1d ops.
    For us, we can treat this as a feed-forward module with matmuls.
    """

    def parameter(key):
        if prefix:
            return parameters[f"{prefix}.{key}"]
        else:
            return parameters[key]

    out = torch.matmul(activations, parameter("c_fc.weight"))
    out = out + parameter("c_fc.bias")
    out = torch.nn.functional.gelu(out)
    out = torch.matmul(out, parameter("c_proj.weight"))
    out = out + parameter("c_proj.bias")
    return out


def functional_gpt2_mha(
    hidden_states, config, parameters, is_qkv_weight_split=False, prefix=None
):
    def parameter(key):
        if prefix:
            return parameters[f"{prefix}.{key}"]
        else:
            return parameters[key]

    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[-2]
    attention_head_size = config.hidden_size // config.n_head

    def permute(x):
        x = x.view(
            [batch_size, seq_len, config.num_attention_heads, attention_head_size]
        )
        return x.permute(0, 2, 1, 3)

    def merge_heads(x):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(
            [batch_size, seq_len, config.num_attention_heads * attention_head_size]
        )

    if is_qkv_weight_split:
        query = torch.matmul(hidden_states, parameter("c_attn.weight.query"))
        query = query + parameter("c_attn.bias.query")

        key = torch.matmul(hidden_states, parameter("c_attn.weight.key"))
        key = key + parameter("c_attn.bias.key")

        value = torch.matmul(hidden_states, parameter("c_attn.weight.value"))
        value = value + parameter("c_attn.bias.value")

    else:
        attention_qkv = torch.matmul(hidden_states, parameter("c_attn.weight"))
        attention_qkv = attention_qkv + parameter("c_attn.bias")
        query, key, value = attention_qkv.split(config.hidden_size, dim=-1)

    query = permute(query)

    key = permute(key)
    key = key.transpose(-1, -2)

    value = permute(value)

    attention_scores = torch.matmul(query, key)

    # Key difference compared to bert mha, we apply a causal mask
    attention_scores = parameter("valid_causal_mask") * attention_scores + parameter(
        "invalid_causal_mask"
    )

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_probs, value)
    attention_output = merge_heads(attention_output)

    attention_output = torch.matmul(attention_output, parameter("c_proj.weight"))
    attention_output = attention_output + parameter("c_proj.bias")
    return attention_output


def functional_gpt2_block(hidden_states, config, parameters, is_qkv_weight_split=False):
    ln1_output = torch.nn.functional.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=parameters["ln_1.weight"],
        bias=parameters["ln_1.bias"],
    )
    attn_output = functional_gpt2_mha(ln1_output, config, parameters, prefix="attn")
    hidden_states_plus_attn_ln = attn_output + hidden_states
    ln_hidden_states_plus_attn_ln = torch.nn.functional.layer_norm(
        hidden_states_plus_attn_ln,
        (hidden_states_plus_attn_ln.shape[-1],),
        weight=parameters["ln_2.weight"],
        bias=parameters["ln_2.bias"],
    )
    feed_forward_hidden_states = functional_gpt2_feedforward(
        ln_hidden_states_plus_attn_ln, parameters, prefix="mlp"
    )
    hidden_states = hidden_states_plus_attn_ln + feed_forward_hidden_states
    return hidden_states


class PyBudaGPT2MLP(PyBudaModule):
    def __init__(self, name, parameters, config, prefix=None):
        super().__init__(name)
        self.parameters = parameters
        self.config = config
        self.prefix = prefix

    def forward(self, activations, gpt_block_index=0, prefix=None):
        def parameter(name):
            if self.prefix:
                return self.parameters[f"{self.prefix}.{name}"]
            else:
                return self.parameters[f"{name}"]

        intermediate = nn.Matmul(
            f"ff_{gpt_block_index}_ff1",
            activations,
            parameter("c_fc.weight"),
            parameter("c_fc.bias"),
        )

        intermediate_gelu = nn.Gelu(f"ff{gpt_block_index}_gelu", intermediate)

        output = nn.Matmul(
            f"ff_{gpt_block_index}_ff2",
            intermediate_gelu,
            parameter("c_proj.weight"),
            parameter("c_proj.bias"),
        )

        return output


class PyBudaGPT2MHA(PyBudaModule):
    def __init__(self, name, parameters, config, prefix=None):
        super().__init__(name)
        self.parameters = parameters
        self.config = config
        self.prefix = prefix

    def forward(self, hidden_states):
        def param(name):
            if self.prefix:
                return self.parameters[f"{self.prefix}.{name}"]
            else:
                return self.parameters[f"{name}"]

        encoder_index = 0

        query = nn.Matmul(
            f"mha_{encoder_index}_query",
            hidden_states,
            param("c_attn.weight.query"),
            param("c_attn.bias.query"),
        )

        query = nn.HSlice(f"mha_{encoder_index}_query_slice", query, self.config.n_head)

        key = nn.Matmul(
            f"mha_{encoder_index}_key",
            hidden_states,
            param("c_attn.weight.key"),
            param("c_attn.bias.key"),
        )

        key = nn.HSlice(f"mha_{encoder_index}_key_slice", key, self.config.n_head)
        key = nn.Transpose(f"mha_{encoder_index}_key_transpose", key, 2, 3)

        value = nn.Matmul(
            f"mha_{encoder_index}_value",
            hidden_states,
            param("c_attn.weight.value"),
            param("c_attn.bias.value"),
        )

        value = nn.HSlice(f"mha_{encoder_index}_value_slice", value, self.config.n_head)

        attention_scores = nn.Matmul(f"mha_{encoder_index}_as", query, key)
        valid_attention_scores = nn.Multiply(
            "valid_attention_scores", attention_scores, param("valid_causal_mask")
        )
        masked_attention_scores = nn.Add(
            "masked_attention_scores",
            valid_attention_scores,
            param("invalid_causal_mask"),
        )

        attention_probs = nn.Softmax(
            f"mha_{encoder_index}_as_softmax", masked_attention_scores, dim=-1
        )

        context = nn.Matmul(f"mha_{encoder_index}_ac", attention_probs, value)
        context = nn.HStack(f"mha_{encoder_index}_ac_stacked", context)

        output = nn.Matmul(
            f"mha_{encoder_index}_output",
            context,
            param("c_proj.weight"),
            param("c_proj.bias"),
        )

        return output


class PyBudaGPT2Block(PyBudaModule):
    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config

        self.attn = PyBudaGPT2MHA("attn", parameters, config, prefix="attn")
        self.mlp = PyBudaGPT2MLP("mlp", parameters, config, prefix="mlp")

    def forward(self, hidden_states):
        def parameter(key):
            return self.parameters[key]

        ln1_output = nn.Layernorm(
            "ln_hidden_states",
            hidden_states,
            parameter("ln_1.weight"),
            parameter("ln_1.bias"),
        )

        attn_output = self.attn(ln1_output)

        hidden_states_plus_attn_ln = nn.Add(
            "hidden_states_plus_attn_ln", attn_output, hidden_states
        )
        ln_hidden_states_plus_attn_ln = nn.Layernorm(
            "ln_hidden_states_plus_attn_ln ",
            hidden_states_plus_attn_ln,
            parameter("ln_2.weight"),
            parameter("ln_2.bias"),
        )

        feed_forward_hidden_states = self.mlp(ln_hidden_states_plus_attn_ln)

        hidden_states = nn.Add(
            "hidden_states_plus_attn_ln_plus_feed_forward_hidden_states",
            hidden_states_plus_attn_ln,
            feed_forward_hidden_states,
        )
        return hidden_states


class PyBudaGPT2LayerNorm(PyBudaModule):
    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config

    def forward(self, hidden_states):
        def parameter(key):
            return self.parameters[key]

        ln1_output = nn.Layernorm(
            "ln_hidden_states",
            hidden_states,
            parameter("ln_1.weight"),
            parameter("ln_1.bias"),
        )

        return ln1_output
