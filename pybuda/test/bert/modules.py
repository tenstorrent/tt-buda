# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Dict

from pybuda import PyBudaModule
from pybuda.op import (
    Matmul,
    HSlice,
    Add,
    Multiply,
    Transpose,
    HStack,
    Gelu,
    Identity,
    Exp,
    Log,
    ReduceSum,
    ReduceAvg,
    Subtract,
    Constant,
)
from pybuda.op import nn
from pybuda import Parameter

def get_bert_parameters(module: str, hidden_dim=128, encoder_index=0, vocab_size=0) -> Dict[str, Parameter]:
    intermed_dim = 4 * hidden_dim
    params = {
        "mha": {
            f"bert.encoder.layer.{encoder_index}.attention.self.query.weight": Parameter(1, 1, hidden_dim, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.self.key.weight": Parameter(1, 1, hidden_dim, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.self.value.weight": Parameter(1, 1, hidden_dim, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.self.query.bias": Parameter(1, 1, 1, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.self.key.bias": Parameter(1, 1, 1, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.self.value.bias": Parameter(1, 1, 1, hidden_dim),
            f"reciprocal_of_sqrt_of_head_size_{encoder_index}": Parameter(1, 1, 1, 1, requires_grad=False),
            f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight": Parameter(1, 1, hidden_dim, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias": Parameter(1, 1, 1, hidden_dim),
        },
        "ff":  {
            f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight": Parameter(1, 1, hidden_dim, intermed_dim),
            f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias": Parameter(1, 1, 1, intermed_dim),
            f"bert.encoder.layer.{encoder_index}.output.dense.weight": Parameter(1, 1, intermed_dim, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.output.dense.bias": Parameter(1, 1, 1, hidden_dim),
        },
        "attn_lnorm": {
            f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight": Parameter(1, 1, 1, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias": Parameter(1, 1, 1, hidden_dim),
        },
        "out_lnorm": {
            f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight": Parameter(1, 1, 1, hidden_dim),
            f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias": Parameter(1, 1, 1, hidden_dim),
        },
        "pred_transform": {
            f"cls.predictions.transform.dense.weight": Parameter(1, 1, hidden_dim, hidden_dim),
            f"cls.predictions.transform.dense.bias": Parameter(1, 1, 1, hidden_dim),
            f"cls.predictions.transform.LayerNorm.weight": Parameter(1, 1, 1, hidden_dim),
            f"cls.predictions.transform.LayerNorm.bias": Parameter(1, 1, 1, hidden_dim),
        },
        "pred_decoder": {
            f"bert.pretrain_head_embeddings.word_embeddings.weight": Parameter(1, 1, hidden_dim, vocab_size)
        },
    }

    if module in params:
        return params[module]

    # combinations
    if module == "ffnorm":
        return {**params["ff"], **params["out_lnorm"]}

    if module == "encoder":
        return {**params["mha"], **params["ff"], **params["out_lnorm"], **params["attn_lnorm"]}

    raise RuntimeError("Unknown bert module type")


class PyBudaBertMHA(PyBudaModule):

    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config

    def forward(self, encoder_input, attention_mask):

        encoder_index = self.config["encoder_index"]
        def param(name):
            return self.parameters[f"bert.encoder.layer.{encoder_index}.attention.{name}"]

        query = Matmul(f"mha_{encoder_index}_query", 
            encoder_input, 
            param("self.query.weight"),
            param("self.query.bias"))

        query = HSlice(f"mha_{encoder_index}_query_slice", query, self.config["num_heads"])

        key = Matmul(f"mha_{encoder_index}_key", 
            encoder_input, 
            param("self.key.weight"),
            param("self.key.bias"))

        key = HSlice(f"mha_{encoder_index}_key_slice", key, self.config["num_heads"])
        key = Transpose(f"mha_{encoder_index}_key_transpose", key, 2, 3)

        value = Matmul(f"mha_{encoder_index}_value", 
            encoder_input, 
            param("self.value.weight"),
            param("self.value.bias"))

        value = HSlice(f"mha_{encoder_index}_value_slice", value, self.config["num_heads"])


        attention_scores = Matmul(f"mha_{encoder_index}_as", query, key)
        attention_scores = Multiply(f"mha_{encoder_index}_as_div", 
                attention_scores,
                self.parameters[f"reciprocal_of_sqrt_of_head_size_{encoder_index}"])
        attention_scores = Add(f"mha_{encoder_index}_as_mask", attention_scores, attention_mask)
        
        attention_probs = nn.Softmax(f"mha_{encoder_index}_as_softmax", attention_scores, dim=-1, stable=True)

        context = Matmul(f"mha_{encoder_index}_ac", attention_probs, value)
        context = HStack(f"mha_{encoder_index}_ac_stacked", context)

        output = Matmul(f"mha_{encoder_index}_output", 
                context,
                param("output.dense.weight"),
                param("output.dense.bias"))

        return output


class PyBudaFeedForward(PyBudaModule):

    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config

    def forward(self, encoder_input):

        encoder_index = self.config["encoder_index"]
        def param(name):
            return self.parameters[f"bert.encoder.layer.{encoder_index}.{name}"]

        intermediate = Matmul(f"ff_{encoder_index}_ff1", 
            encoder_input, 
            param("intermediate.dense.weight"),
            param("intermediate.dense.bias"))

        intermediate_gelu = Gelu(f"ff{encoder_index}_gelu", intermediate)

        output = Matmul(f"ff_{encoder_index}_ff2", 
            intermediate_gelu, 
            param("output.dense.weight"),
            param("output.dense.bias"))

        return output

class PyBudaFFNorm(PyBudaModule):

    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config
        self.ff = PyBudaFeedForward("ff", parameters, config)

    def forward(self, input):

        encoder_index = self.config["encoder_index"]
        def param(name):
            return self.parameters[f"bert.encoder.layer.{encoder_index}.{name}"]

        ff = self.ff(input)

        result = nn.Layernorm("norm_ff", Add("add_ff", input, ff), 
            param("output.LayerNorm.weight"), param("output.LayerNorm.bias"))

        return result


class PyBudaBertEncoder(PyBudaModule):

    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config
        self.mha = PyBudaBertMHA("mha", parameters, config)
        self.ff = PyBudaFeedForward("ff", parameters, config)

    def forward(self, encoder_input, attention_mask):

        encoder_index = self.config["encoder_index"]
        def param(name):
            return self.parameters[f"bert.encoder.layer.{encoder_index}.{name}"]

        mha = self.mha(encoder_input, attention_mask)
        output = nn.Layernorm(f"norm_mha_{encoder_index}", Add(f"add_mha_{encoder_index}", encoder_input, mha), 
            param("attention.output.LayerNorm.weight"), param("attention.output.LayerNorm.bias"))

        ff = self.ff(output)

        result = nn.Layernorm(f"norm_ff_{encoder_index}", Add(f"add_ff_{encoder_index}", output, ff), 
            param("output.LayerNorm.weight"), param("output.LayerNorm.bias"))

        if self.config.get("passthrough_attn_mask", None):
            return result, attention_mask
        else:
            return result

class PyBudaPredictionHeadTransform(PyBudaModule):
    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config

    def forward(self, hidden_states):

        def param(name):
            return self.parameters[f"cls.predictions.transform.{name}"]

        hidden_states = Matmul("pred_matmul", hidden_states, param("dense.weight"))
        hidden_states = Add("pred_bias", hidden_states, param("dense.bias"))
        hidden_states = Gelu("pred_gelu", hidden_states)
        hidden_states = nn.Layernorm("pred_ln", hidden_states, param("LayerNorm.weight"), param("LayerNorm.bias"))
        return hidden_states

class PyBudaPredictionHeadDecoder(PyBudaModule):
    def __init__(self, name, parameters, config):
        super().__init__(name)
        self.parameters = parameters
        self.config = config

    def forward(self, hidden_states):

        def param(name):
            return self.parameters[f"bert.pretrain_head_embeddings.word_embeddings.{name}"]

        return Matmul("pred_decoder_matmul", hidden_states, param("weight"))

