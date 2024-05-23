# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
import torch.nn as nn

from transformers import GPTJConfig
from transformers.models.gptj.modeling_gptj import GPTJBlock

from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendDevice,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.op.eval import compare_tensor_to_golden
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import pybuda

def test_gptj_block(test_kind, test_device):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()
    input_shape = (1, 128, 4096)
    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    if test_device.arch == BackendDevice.Wormhole_B0 or test_device.arch == BackendDevice.Blackhole:
        pytest.skip() # see tenstorrent/pybuda#969

    #Fusing disabled due to tenstorrent/pybuda#789
    if (test_kind == TestKind.INFERENCE):
        compiler_cfg.enable_auto_fusing=False

    config = GPTJConfig(n_layer=1)  # for faster loading
    config.rotary_dim = 64
    model = GPTJBlock(config)

    mod = PyTorchModule("gptj_block", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq)
        .to(x.device)
        .float()
    )

    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)

    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(
            2, 3
        ),
        sincos,
    )

    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)



def test_tvm_rotate_every_two(test_kind, test_device): 
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    input_shape = (1, 128, 16, 256)
    compiler_cfg = _get_global_compiler_config()

    class GPTJRotateEveryTwo(nn.Module):
        def __init__(self, config):
            super().__init__()

        def forward(self, key):
            seq_len = key.shape[1]
            k_rot = key[:, :, :, :64]
            k_pass = key[:, :, :, 64:]
            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=0)
            key = torch.cat([k_rot, k_pass], dim=-1)

            return key

    config = GPTJConfig()
    model = GPTJRotateEveryTwo(config)
    mod = PyTorchModule("fixed_pos_embedding", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
