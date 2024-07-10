# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os
import pytest

import onnx
import torch
import torch.nn as nn

from transformers import GPTJConfig
from transformers.models.gptj.modeling_gptj import GPTJBlock

from pybuda import (
    OnnxModule,
    CompileDepth,
    VerifyConfig,
)

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind





def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len).float(), inv_freq)
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
    map_sincos = []
    for val in sincos:
        re_val = val[None, offset : x.shape[1] + offset, None, :]
        # re_val = val[None, :, None, :]
        # re_val = val.reshape(1, val.shape[-2], 1, val.shape[-1])
        rep_val = re_val.repeat_interleave(2, 3)
        map_sincos.append(rep_val)
    sin, cos = map_sincos

    return (x * cos) + (rotate_every_two(x) * sin)


def test_tvm_rotate_every_two(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class GPTJRotateEveryTwo(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, key):
            seq_len = key.shape[1]
            k_rot = key[:, :, :, :64]
            k_pass = key[:, :, :, 64:]
            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=0)
            key = torch.cat([k_rot, k_pass], dim=-1)

            return key

    class GPTJRotateEveryTwoWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.rand((1, 128, 16, 256))
            self.module = GPTJRotateEveryTwo()

        def forward(self, x):
            return self.module(self.x) + x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL

    # Configure PyTorch module
    pytorch_module = GPTJRotateEveryTwoWrapper()

    # Export to ONNX
    input_shape = (1, 128, 16, 256)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/rotate_every_two.onnx"
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "rotate_every_two_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    # Cleanup
    os.remove(save_path)
