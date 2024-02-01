# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# port of models described in RW
# We use the bloom model as a starting point for these model.
# Please refer to the bloom models for usage instructions.

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_RW import RWConfig

logger = logging.get_logger(__name__)

# NOTE(Hesslow): Unfortunately we did not fuse matmul and bias during training, this means that there's one additional quantization to bfloat16 between the operations.
# In order not to degrade the quality of our HF-port, we keep these characteristics in the final model.
class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


from einops import rearrange


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def gather_cos_sin(cos, sin, position_ids, batch_size=1):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(batch_size, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin have already been gathered
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbeddingTT(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#        freqs = torch.mul(t.view((t.shape[0], 1)), self.inv_freq.view((1, self.inv_freq.shape[0]))) # einsum free implementation
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class TT_functional:
    def scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False):
        DTYPE = Q.dtype
        L, S = Q.size(-2), K.size(-2)

        def make_mask(L, S, DTYPE):
            attn_mask = torch.ones(L, S, dtype=DTYPE).tril(diagonal=0).to(K.device)
            inverted_mask = 1.0 - attn_mask
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(DTYPE).min)

        assert is_causal or attn_mask is not None, "attn_mask must be provided if is_causal is False"
        assert not is_causal or attn_mask is None, "attn_mask must be None if is_causal is True"

        if attn_mask is None or is_causal:
            attn_mask = make_mask(L, S, DTYPE)

        #attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(Q.size(-1), dtype=DTYPE))) + attn_mask, dim=-1)
        #attn_weight = torch.dropout(attn_weight, dropout_p, train)
        ATT = Q @ K.transpose(-2, -1) / torch.tensor(Q.size(-1)**(1/2), dtype=DTYPE).to(K.device)
        attn_weight = F.softmax(ATT + attn_mask, dim=-1, dtype=DTYPE)
        attn_weight = nn.Dropout(p=dropout_p)(attn_weight)
        return attn_weight @ V

class PaddedAttentionTT(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()

        self.original_hidden_size = config.hidden_size
        self.head_dim = self.original_hidden_size // config.n_head
        self.hidden_size = config.hidden_size + config.padded_heads*self.head_dim
        self.num_heads = config.n_head + config.padded_heads
        self.padded_heads = config.padded_heads
        self.pad_amt = self.padded_heads*self.head_dim
        self.hidden_dropout = config.hidden_dropout
        self.mask_lora_activations = config.mask_lora_activations

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.wq = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.wk = Linear(self.hidden_size, self.head_dim, bias=config.bias)
        self.wv = Linear(self.hidden_size, self.head_dim, bias=config.bias)

        self.multi_query = config.multi_query
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head if not self.multi_query else 1

        self.init_padding = False
        self.did_split = False

        #Creating a mask for zeroing out padded hidden states
        self.lora_hidden_states_mask = torch.zeros((1, config.max_sequence_length, self.hidden_size))
        self.lora_hidden_states_mask[...,:self.original_hidden_size] = 1.0
        self.lora_hidden_states_mask[...,self.original_hidden_size:] = 0.0

    def make_pad_weights(self, make_kvs=False):
        """
        mask weight of wq and dense so that forward preserves padding
        """
        self.wq.weight.data = self.mask_weight(self.wq.weight.data, self.original_hidden_size, self.original_hidden_size)
        self.dense.weight.data = self.mask_weight(self.dense.weight.data, self.original_hidden_size, self.original_hidden_size)
        if make_kvs:
            # by default, this should not be necessary as input activation has padded features 0
            self.wk.weight.data = self.mask_weight(self.wk.weight.data, self.head_dim, self.original_hidden_size)
            self.wv.weight.data = self.mask_weight(self.wv.weight.data, self.head_dim, self.original_hidden_size)

        self.init_padding = True
        

    def mask_weight(self, w: torch.Tensor, c, d) -> torch.Tensor:
        """
        for a given tensor of shape (..., a, b), lets look at the last 2 dimensions in blocks
        [A, B;
         C, D]                 We want to keep A, and make B, C, D be 0. A is shape (c,d)
        """

        # Create a mask tensor with ones for the A block and zeros for the other blocks
        mask = torch.zeros_like(w)
        mask[..., :c, :d] = 1.0
        mask[..., :c, d:] = 0.0
        mask[..., c:, :] = 0.0

        # Apply the mask to the input tensor
        masked_w = w * mask

        return masked_w

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ):
        

        assert self.init_padding, "Attention weights have not been split yet. Call `split_qkv_weights()` before using this layer."
        assert self.did_split

        # query_layer: [batch, seqlen, num_heads, head_dim]
        query_layer = self.wq(hidden_states)
        
        if self.mask_lora_activations:
            # Mask q projection
            query_layer = query_layer*self.lora_hidden_states_mask.to(query_layer.device)
        query_layer = query_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)

        #No need to pad key value projections as they are single headed only
        key_layer = self.wk(hidden_states)
        value_layer = self.wv(hidden_states)

        if self.multi_query:
            key_layer = key_layer.unsqueeze(2)      # [batch_size, seq_length, 1, head_dim]
            value_layer = value_layer.unsqueeze(2)  # [batch_size, seq_length, 1, head_dim]

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2)   # [batch_size, num_heads, seq_length, head_dim]
        key_layer = key_layer.transpose(1, 2)       # [batch_size, 1, seq_length, head_dim]
        value_layer = value_layer.transpose(1, 2)   # [batch_size, 1, seq_length, head_dim]


        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
        
        is_causal = True
        attention_mask = None

        attn_output = TT_functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal=is_causal
        )

        x = attn_output.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        return output_tensor

    def split_qkv_weights(self):

        # load wq, wk, wv from query_key_value
        self.wq.weight = nn.Parameter(self.query_key_value.weight[:self.hidden_size].clone())

        self.wk.weight = nn.Parameter(self.query_key_value.weight[self.hidden_size : self.hidden_size + self.head_dim].clone())

        self.wv.weight = nn.Parameter(self.query_key_value.weight[self.hidden_size + self.head_dim :].clone())

        self.did_split = True


class Attention(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.use_cache = config.use_cache

        self.did_split = False

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # self.maybe_rotary = RotaryEmbedding(config.head_dim) if config.rotary else lambda q, k: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = Linear(
            self.hidden_size,
            3 * self.hidden_size if not config.multi_query else (self.hidden_size + 2 * self.head_dim),
            bias=config.bias,
        )

        self.wq = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.wk = Linear(self.hidden_size, self.head_dim, bias=config.bias)
        self.wv = Linear(self.hidden_size, self.head_dim, bias=config.bias)

        self.multi_query = config.multi_query
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head if not self.multi_query else 1

    def split_qkv_weights(self):

        # load wq, wk, wv from query_key_value
        self.wq.weight = nn.Parameter(self.query_key_value.weight[:self.hidden_size].clone())

        self.wk.weight = nn.Parameter(self.query_key_value.weight[self.hidden_size : self.hidden_size + self.head_dim].clone())

        self.wv.weight = nn.Parameter(self.query_key_value.weight[self.hidden_size + self.head_dim :].clone())

        self.did_split = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cos=None,
        sin=None
    ):
        
        assert self.did_split, "Attention weights have not been split yet. Call `split_qkv_weights()` before using this layer."

        # query_layer: [batch, seqlen, num_heads, head_dim]
        query_layer = self.wq(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
        key_layer = self.wk(hidden_states)
        value_layer = self.wv(hidden_states)

        if self.multi_query:
            key_layer = key_layer.unsqueeze(2)      # [batch_size, seq_length, 1, head_dim]
            value_layer = value_layer.unsqueeze(2)  # [batch_size, seq_length, 1, head_dim]

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2)   # [batch_size, num_heads, seq_length, head_dim]
        key_layer = key_layer.transpose(1, 2)       # [batch_size, 1, seq_length, head_dim]
        value_layer = value_layer.transpose(1, 2)   # [batch_size, 1, seq_length, head_dim]



        # query_layer, key_layer = self.maybe_rotary(query_layer, key_layer)
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        key_layer_ret, value_layer_ret = key_layer.squeeze(2), value_layer.squeeze(2)

        if layer_past is not None and layer_past[0] is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            past_key = past_key.view(batch_size, 1, -1, self.head_dim)
            past_value = past_value.view(batch_size, 1, -1, self.head_dim)
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        _, _, kv_length, _ = key_layer.shape

        if layer_past is not None and layer_past[0] is not None:
            assert q_length == 1, "Input can only have one token if we're passing in a layer_past"
            # attention_mask = torch.ones(1, kv_length, dtype=torch.bool)
            is_causal = False
        else:
            is_causal = True
            attention_mask = None

        # if self.use_cache:
        #     present = (key_layer.reshape(batch_size, kv_length, self.head_dim), value_layer.reshape(batch_size, kv_length, self.head_dim))
        # else:
        #     present = None

        attn_output = TT_functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal=is_causal
        )

        # x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = attn_output.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        # outputs = (output_tensor, present) if present is not None else output_tensor

        # return output_tensor, present[0], present[1]
        return output_tensor, key_layer_ret, value_layer_ret


class MLP(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x

class PaddedMLP(nn.Module):
    """
    Padded MLP layer, assume no bias
    Must call make_pad_weights() after loading new weights in.
    """
    def __init__(self, config: RWConfig):
        super().__init__()
        self.original_hidden_size = config.hidden_size
        self.head_dim = self.original_hidden_size // config.n_head
        hidden_size = config.hidden_size + config.padded_heads*self.head_dim
        self.num_heads = config.n_head + config.padded_heads
        self.padded_heads = config.padded_heads
        self.pad_amt = self.padded_heads*self.head_dim

        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

        self.init_padding = False
        
    def make_pad_weights(self):
        """
        mask weight of dense_4h_to_h and dense_h_to_4h so that forward preserves padding
        """
        self.dense_4h_to_h.weight.data = self.mask_weight(self.dense_4h_to_h.weight.data, self.original_hidden_size, self.original_hidden_size*4)
        self.dense_h_to_4h.weight.data = self.mask_weight(self.dense_h_to_4h.weight.data, self.original_hidden_size*4, self.original_hidden_size)

        self.init_padding = True

    def mask_weight(self, w: torch.Tensor, c, d) -> torch.Tensor:
        """
        for a given tensor of shape (..., a, b), lets look at the last 2 dimensions in blocks
        [A, B;
         C, D]                 We want to keep A, and make B, C, D be 0. A is shape (c,d)
        """
        # Extract the shape of the tensor and the number of blocks
        shape = w.size()

        # Create a mask tensor with ones for the A block and zeros for the other blocks
        mask = torch.zeros_like(w)
        mask[..., :c, :d] = 1.0
        mask[..., :c, d:] = 0.0
        mask[..., c:, :] = 0.0

        # Apply the mask to the input tensor
        masked_w = w * mask

        return masked_w

    def forward(self, x: torch.Tensor, masked_input=False) -> torch.Tensor:
        assert self.init_padding, "Must call make_pad_weights() after loading new weights in."
        x = self.act(self.dense_h_to_4h(x)) 
        x = self.dense_4h_to_h(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = Attention(config)
        self.use_cache = config.use_cache

        self.mlp = MLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

        self.config = config

        self.did_split = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ):

        layernorm_output = self.input_layernorm(hidden_states)
        # residual = hidden_states
        hidden_states = hidden_states.to(self.input_layernorm.mask.device)
        residual = hidden_states*self.input_layernorm.mask # padding requires this

        # Self attention.
        attention_output = self.self_attention(
            layernorm_output,
            cos=cos,
            sin=sin
        )

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        return output
    
    def pad_decoder(self):

        assert self.self_attention.did_split, "Must call split_heads() before padding decoder"

        self.num_heads = self.config.n_head + self.config.padded_heads
        self.padded_heads = self.config.padded_heads
        self.head_dim = self.config.hidden_size // self.config.n_head
        self.pad_amt = self.padded_heads*self.head_dim

        unpadded_hidden = self.config.hidden_size

        padded_hidden_size = self.config.hidden_size + self.pad_amt
        assert padded_hidden_size % (self.num_heads) == 0, "hidden_size must be divisible by n_head+padded_heads"

        # replace layernorm with padded
        padded_ln = PaddedLayerNorm(padded_hidden_size, eps=self.config.layer_norm_epsilon, pad_amt=self.pad_amt)
        
        padded_ln.weight.data[:unpadded_hidden] = self.input_layernorm.weight.data
        padded_ln.bias.data[:unpadded_hidden] = self.input_layernorm.bias.data
        self.input_layernorm = padded_ln

        # replace mlp with padded
        padded_mlp = PaddedMLP(self.config)
        padded_mlp.dense_h_to_4h.weight.data[:4*unpadded_hidden,:unpadded_hidden] = self.mlp.dense_h_to_4h.weight.data
        padded_mlp.dense_4h_to_h.weight.data[:unpadded_hidden,:4*unpadded_hidden] = self.mlp.dense_4h_to_h.weight.data
        padded_mlp.make_pad_weights()
        self.mlp = padded_mlp

        # replace attention with padded
        padded_attn = PaddedAttentionTT(self.config)
        padded_attn.wq.weight.data[:unpadded_hidden,:unpadded_hidden] = self.self_attention.wq.weight.data
        padded_attn.dense.weight.data[:unpadded_hidden,:unpadded_hidden] = self.self_attention.dense.weight.data
        padded_attn.wk.weight.data[:, :unpadded_hidden] = self.self_attention.wk.weight.data
        padded_attn.wv.weight.data[:, :unpadded_hidden] = self.self_attention.wv.weight.data
        padded_attn.make_pad_weights(make_kvs=True)
        padded_attn.did_split = self.self_attention.did_split
        self.self_attention = padded_attn




class RWPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RWConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, RWModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


class SequentialCaller(nn.Module):
    def __init__(self, layers): #, norm, lm_head):
        super().__init__() 
        self.layers = layers

    def forward(self, hidden_states, cos, sin):

        for i, block in enumerate(self.layers):

            hidden_states = block(
                hidden_states,
                cos=cos,
                sin=sin
            )

        return hidden_states
    
    
class PaddedLayerNorm(nn.Module):
    """
    Padded layernorm
    Supports n_features dimension on (n_batch, n_seq, n_features) input shape
    Assume padding at the end of the dimension
    Training not yet supported
    """
    def __init__(self, normalized_shape, pad_amt,eps=1e-05, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.pad_amt = pad_amt
        self.mask = torch.ones(normalized_shape[0])
        self.mask[-pad_amt:] = 0
        self.scale = normalized_shape[0]/(normalized_shape[0]-pad_amt) # this is the scale factor for the mean, n_p/n

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, masked_input=False):
        self.mask = self.mask.to(self.weight.device)
        input = input.to(self.weight.device)
        # assume input is [batch_size, seq_length, num_heads*head_dim]
        if not masked_input:
            x_p = input * self.mask # this masks out the padded values, if we can assure input is already 0 padded, we can remove this
        else:
            x_p = input
        mean = self.scale*torch.mean(x_p, dim=-1, keepdim=True)
        x_p2 = self.scale*torch.mean(x_p**2, dim=-1, keepdim=True)
        variance = x_p2 - torch.square(mean)# assume unbiased=False
        normalized_input = (x_p - mean) / torch.sqrt(variance + self.eps)

        # fracture multiply add
        if self.elementwise_affine:
            return (self.weight * normalized_input + self.bias)*self.mask
        return normalized_input * self.mask


class BeforeDecoderModule(nn.Module):
    def __init__(self, config, word_embeddings) -> None:
        super().__init__()
        self.config = config
        # Embedding + LN Embedding
        self.word_embeddings = word_embeddings
        self.rotary_emb = RotaryEmbeddingTT(config.head_dim, 2048, 10000, None)

    def forward(self, input_ids, past_hidden_states=None):
        
        if past_hidden_states is None:
            inputs_embeds =  self.word_embeddings(input_ids)
            hidden_states = inputs_embeds

            # Pad inputs to model
            padded_hidden = self.config.hidden_size + (self.config.hidden_size // self.config.n_head) * self.config.padded_heads
            hidden_states_padded = torch.zeros((hidden_states.size(0), hidden_states.size(1), padded_hidden))
            hidden_states_padded[:,:,:self.config.hidden_size] = hidden_states
            hidden_states = hidden_states_padded
        else:

            hidden_states = past_hidden_states.to(input_ids.device)

        # Rotary Embeddings
        batch_size, seq_length, _ = hidden_states.shape
        cos, sin = self.rotary_emb()
        position_ids = torch.arange(seq_length).unsqueeze(0).to(cos.device)
        cos, sin = gather_cos_sin(cos, sin, position_ids, batch_size=batch_size)

        return hidden_states, cos, sin

class RWModel(RWPreTrainedModel):
    def __init__(self, config: RWConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        decoder_start_idx = 0
        decoder_end_idx = config.num_hidden_layers
        
        if config.cache_hidden_states and not config.load_hidden_states:
            decoder_end_idx = config.num_cache_layers if config.num_hidden_layers>=config.num_cache_layers else config.num_hidden_layers
        elif not config.cache_hidden_states and config.load_hidden_states:
            decoder_start_idx = config.num_cache_layers if config.num_hidden_layers>=config.num_cache_layers else config.num_hidden_layers

        self.blocks = SequentialCaller(self.h[decoder_start_idx:decoder_end_idx])
        
        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()
    
    def pad_decoders(self):
        for h in self.h:
            h.pad_decoder()

    def split_qkv_weights(self):
        for h in self.h:
            h.self_attention.split_qkv_weights()
    

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        hidden_states,
        cos,
        sin
    ) -> torch.Tensor:
                
        hidden_states = self.blocks(hidden_states, cos, sin)

        return hidden_states
        

class AfterDecoderModule(nn.Module):
    def __init__(self, config, ln_f, lm_head) -> None:
        super().__init__()
        self.config = config
        self.ln_f = ln_f
        self.lm_head = lm_head
        
    def forward(self, hidden_states):

        # Unpad outputs from model
        hidden_states = hidden_states[:,:,:self.config.hidden_size]

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        return lm_logits


class CausalLMLoss(nn.Module):
    def __init__(self, ignore_pad_tokens=False, pad_token_id=11) -> None:
        super().__init__()
        self.loss_fct = CrossEntropyLoss(ignore_index=pad_token_id if ignore_pad_tokens else -100)
        self.loss_scale = 1.0

    def forward(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss = self.loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
        )
        loss = loss * self.loss_scale
        return loss

    def set_loss_scale(self, loss_scale):
        self.loss_scale = loss_scale



class RWForCausalLM(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.cache_hidden_states = config.cache_hidden_states
        self.transformer = RWModel(config)
        self.before_decoders = BeforeDecoderModule(config, self.transformer.word_embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.after_decoders = AfterDecoderModule(config, self.transformer.ln_f, self.lm_head)
        self.loss_fct = CausalLMLoss(ignore_pad_tokens=config.ignore_pad_tokens, pad_token_id=config.pad_token_id)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids=None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hidden_states, cos, sin = self.before_decoders(input_ids, past_hidden_states)
        
        hidden_states = self.transformer(hidden_states, cos, sin)

        lm_logits = self.after_decoders(hidden_states)

        
        if labels is None:
            labels = input_ids
            
        loss = self.loss_fct(lm_logits, labels)
        
        if not return_dict:
            output = (lm_logits,) + None
            return ((loss,) + output)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=hidden_states if self.cache_hidden_states else None,
            attentions=None,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_rw_cache(reordered_past)
