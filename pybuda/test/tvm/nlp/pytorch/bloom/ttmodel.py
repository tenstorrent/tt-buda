# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import math
import torch
from torch import nn
import torch.nn.functional as F
from argparse import Namespace


def tinybloom_model_args(dropout=True):
    """ Just the tinyBloom arguments relevant to model definition, excluding 
        training, data processing and so on. Unsupported options from original 
        codebase are elided / hard-coded """
    args = Namespace(num_layers=2, hidden_size=128, ffn_hidden_size=512, 
                     num_attention_heads=1, kv_channels=128, seq_length=512, 
                     micro_batch_size=16, params_dtype=torch.float32,
                     attention_dropout=0.1, hidden_dropout=0.1, 
                     init_method_std=0.0048, layernorm_epsilon=1e-05,
                    )
    if not dropout:
        args.attention_dropout = 0.
        args.hidden_dropout = 0.
    return args

# Settings used globally within this file
args = tinybloom_model_args(dropout=False)
init_method = lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=args.init_method_std)
output_layer_init_method = lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, 
                                                                std=args.init_method_std / math.sqrt(2.0 * args.num_layers))

class Transformer(nn.Module):
    """ Exact reimplementation of the Bloom ParallelTransformer class.
        This version is much simpler because all the parallelism and
        optimization options are elided / hard-coded for our use case. """
    def __init__(self, transpose_hidden_states=True):
        super().__init__()
        self.transpose_hidden_states = transpose_hidden_states
        self.layers = nn.ModuleList([TransformerLayer(i+1, transpose_hidden_states) for i in range(args.num_layers)])
        self.final_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)
        
    def forward(self, hidden_states):
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        if self.transpose_hidden_states:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.transpose_hidden_states)

        # Reverting data format change [s b h] --> [b s h].
        if self.transpose_hidden_states:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        output = self.final_layernorm(hidden_states)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, layer_number, transpose_hidden_states):
        super().__init__()
        assert layer_number >= 1, "layer_number counts from 1"
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, elementwise_affine=True)
        self.self_attention = Attention(args, layer_number)
        self.post_attention_dropout = nn.Dropout(args.hidden_dropout)
        self.post_attention_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, elementwise_affine=True)
        self.mlp = MLP()
        self.post_mlp_dropout = nn.Dropout(args.hidden_dropout)
        self.alibi = build_alibi_tensor(args.seq_length, args.num_attention_heads, args.micro_batch_size, transpose_hidden_states)
      
    def forward(self, hidden_states, transpose_hidden_states):
        """ Unlike BigScience's implementation we infer the attention mask from the sequence length 
            instead of taking a mask as an input (so the where op can be constant folded) """
        residual = hidden_states
        
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(layernorm_output, self.alibi, transpose_hidden_states)
        attention_output = self.post_attention_dropout(attention_output)
        
        layernorm_input = residual + attention_output
        
        residual = layernorm_input
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        mlp_output = self.post_mlp_dropout(mlp_output)

        output = residual + mlp_output
        return output
        

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_h_to_4h = ZeroBiasLinear(args.hidden_size    , args.ffn_hidden_size, init_method             )
        self.dense_4h_to_h = ZeroBiasLinear(args.ffn_hidden_size, args.hidden_size    , output_layer_init_method)

    def forward(self, x):
        h = self.dense_h_to_4h(x)
        h = openai_gelu(h)
        o = self.dense_4h_to_h(h)
        return o       


class ZeroBiasLinear(nn.Module):
    """ Used instead of nn.Linear to exactly match the pseudorandom generator 
        calls made in the BigScience codebase and ensure identical 
        initialization for the same seed """
    def __init__(self, in_features, out_features, init_method):
        super().__init__()        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=args.params_dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=args.params_dtype))

        init_method(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, args, layer_number):
        super().__init__()
        assert layer_number >= 1, "layer_number counts from 1"
        self.layer_number = layer_number
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size_per_attention_head = args.kv_channels
        
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head) * self.layer_number
        self.projection_size = self.hidden_size_per_attention_head * args.num_attention_heads

        self.query_key_value = ZeroBiasLinear(args.hidden_size, 3 * self.projection_size, init_method)
        self.scale_mask_softmax = ScaleMaskSoftmax(
            softmax_in_fp32=True,
            scale=self.layer_number)
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)
        self.dense = ZeroBiasLinear(self.projection_size, args.hidden_size, output_layer_init_method)
        
    def forward(self, hidden_states, alibi, transpose_hidden_states):
        if not transpose_hidden_states:
            mixed_x_layer = self.query_key_value(hidden_states)

            (query_layer,
            key_layer,
            value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

            new_shape = query_layer.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
            query_layer = query_layer.view(new_shape)
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = key_layer.view(new_shape)
            key_layer = key_layer.permute(0, 2, 1, 3)
            value_layer = value_layer.view(new_shape)
            value_layer = value_layer.permute(0, 2, 1, 3)

            output_size = (query_layer.size(0),
                        query_layer.size(1),
                        query_layer.size(2),
                        key_layer.size(2))

            matmul_result = alibi[:query_layer.shape[0], :, :, :query_layer.shape[2]]
            
            beta  = 1.0 / self.layer_number
            alpha = 1.0 / self.norm_factor
            matmul_result = beta * matmul_result + alpha * torch.matmul(query_layer, key_layer.transpose(-1, -2))

            attention_scores = matmul_result.view(*output_size)

            attention_probs = self.scale_mask_softmax(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_attention_head * self.num_attention_heads, )
            context_layer = context_layer.view(new_context_layer_shape)

            output = self.dense(context_layer)
        else:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads,
                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
            key_layer,
            value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            
            # ===================================
            # Raw attention scores. [b, np, s, s]
            # ===================================

            # [b, np, sq, sk]
            output_size = (query_layer.size(1),
                        query_layer.size(2),
                        query_layer.size(0),
                        key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2],
                                        output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3],
                                    output_size[0] * output_size[1], -1)

            # preallocting result tensor: [b * np, sq, sk]
            matmul_result = alibi[:output_size[0]*output_size[1], :, :output_size[3]]
            
            # manual baddbmm as this isn't supported by pybuda yet
            beta  = 1.0 / self.layer_number
            alpha = 1.0 / self.norm_factor
            matmul_result = beta * matmul_result + alpha * torch.bmm(
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                )
            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)
            
            # attention scores and attention mask [b, np, sq, sk]
            attention_probs = self.scale_mask_softmax(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)

            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1),
                        value_layer.size(2),
                        query_layer.size(0),
                        value_layer.size(3))

            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0),
                                        output_size[0] * output_size[1], -1)

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                                output_size[2], -1)

            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + \
                (self.projection_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            # =================
            # Output. [sq, b, h]
            # =================
            output = self.dense(context_layer)
        return output
        

class ScaleMaskSoftmax(torch.nn.Module):
    """
    scaling + mask + softmax

    Arguments:
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """
    def __init__(
        self,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_float16 = False
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        # Unlike BigScience's implementation we take a slice of a constant mask instead of passing it
        # as an additional input because Buda needs use constant folding to handle where operators
        self.mask = torch.tril(torch.ones((args.seq_length, args.seq_length), dtype=torch.uint8)).view(
                1, 1, args.seq_length, args.seq_length)
       
        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, attention_scores):
        # [b, np, sq, sk]
        assert attention_scores.dim() == 4
        
        if self.input_in_float16 and self.softmax_in_fp32:
            attention_scores = attention_scores.float()

        attention_scores = attention_scores * self.scale

        query_length, key_length = attention_scores.size(-2), attention_scores.size(-1)
        causal_mask = self.mask[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(-10000.0, dtype=attention_scores.dtype).to(attention_scores.device)
        attention_scores = torch.where(causal_mask, attention_scores, mask_value)

        # Original implementation: attention_scores.masked_fill_(attention_mask, -10000.0)
        probs = torch.nn.Softmax(dim=-1)(attention_scores)

        if self.input_in_float16 and self.softmax_in_fp32:
            probs = probs.bfloat16()

        return probs


@torch.jit.script
def openai_gelu(x):
    """OpenAI's gelu implementation, used by Bloom when bias_gelu_fusion is true even if openai_gelu is false."""
    return torch.nn.functional.gelu(x)
    # return 0.5 * x * (1.0 + torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))


def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = torch.div(tensor.size()[last_dim], num_partitions, rounding_mode='floor')
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    return tensor_list


def build_alibi_tensor(max_seq_len, num_attention_heads, batch_size, transpose_hidden_states):
    # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(num_attention_heads))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
        num_attention_heads, -1, -1)

    if transpose_hidden_states:
        alibi = alibi.repeat(batch_size, 1, 1)
    else:
        alibi = alibi.repeat(batch_size, 1, 1, 1)
    return alibi
