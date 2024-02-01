# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from pybuda.config import CompileDepth
import pytest

import torch
import torch.nn as nn
from transformers.models.xlm import XLMConfig, XLMModel, XLMPreTrainedModel

import math
import itertools
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    tvm_to_python,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind



class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, 1e-10)  # (bs, n_heads, qlen, klen)

        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class XLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index

        # model parameters
        self.dim = config.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads  # 8 by default
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        self.attention = MultiHeadAttention(self.n_heads, self.dim, config=config)
        self.ln = nn.LayerNorm(self.dim, eps=config.layer_norm_eps)


    def get_masks(self, slen, lengths, causal, padding_mask=None):
        """
        Generate hidden states mask, and optionally an attention mask.
        """
        alen = torch.arange(slen, dtype=torch.long)
        if padding_mask is not None:
            mask = padding_mask
        else:
            assert lengths.max().item() <= slen
            mask = alen < lengths[:, None]

        # attention mask is the same as mask, or triangular inferior attention (causal)
        bs = lengths.size(0)
        if causal:
            attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
        else:
            attn_mask = mask

        # sanity check
        assert mask.size() == (bs, slen)
        assert causal is False or attn_mask.size() == (bs, slen, slen)

        return mask, attn_mask

    def forward(self, hidden_states):
        bs, slen = hidden_states.shape[0], hidden_states.shape[1]
        lengths = torch.tensor([slen] * bs)

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        mask, attn_mask = self.get_masks(slen, lengths, self.causal)
        attn_outputs = self.attention(
            hidden_states,
            attn_mask,
        )
        return self.ln(attn_outputs[0])

input_shapes = [(1, 16, 2048)]

def test_tvm_xlm_attention(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    config = XLMConfig()

    model = XLMAttention(config)

    mod = PyTorchModule("XLM_attention", model)

    input_shape = (1, 16, 2048)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"attention.k_lin.bias"},
        ),
        uniform_inputs=True,
    )

def test_tvm_xlm_FFN(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    recompute = True

    config = XLMConfig()

    model = XLMModel(config)

    mod = PyTorchModule("XLM_FFN", model.ffns[0])

    input_shape = (1, 16, 2048)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
