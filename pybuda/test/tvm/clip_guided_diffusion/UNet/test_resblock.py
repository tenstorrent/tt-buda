# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from time import time
from pybuda.verify.backend import verify_module
import pytest

import torch
import torch.nn as nn
from loguru import logger
import pybuda
from pybuda.config import CompileDepth

from pybuda import (
    TTDevice,
    BackendType,
    pybuda_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
    pybuda_compile
)


from test.tvm.utils import evaluate_framework_vs_pybuda

from test.legacy_tests.clip_guided_diffusion.unet.pytorch_unet import ResBlock
from pybuda.config import _get_global_compiler_config

def init_resblock(
    ch, out_channels, upsample=False, downsample=False, use_layer_norm=False, time_embed_dim = 1024
):
    dims = 2
    dropout = 0.0
    time_embed_dim = time_embed_dim

    use_checkpoint = False
    use_scale_shift_norm = True

    return ResBlock(
        ch,
        time_embed_dim,
        dropout,
        out_channels=out_channels,
        dims=dims,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        up=upsample,
        down=downsample,
        use_layer_norm=use_layer_norm,
    )


def test_tvm_resblock(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    t_embed = 64
    ch = 32
    out_channels = 32
    model = init_resblock(ch=ch, out_channels=out_channels, time_embed_dim=t_embed)
    mod = PyTorchModule("ResBlock", model)
    
    input_shape = (1, ch, ch, ch)
    embed_shape = (1, t_embed)

    verify_module(
        mod,
        (input_shape, embed_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            intermediates=True,
            verify_last=False
        ),
        uniform_inputs=True,
    )


def test_tvm_upsample_resblock(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    model = init_resblock(64, 64, time_embed_dim=64, upsample=True)
    mod = PyTorchModule("ResBlockUpsample", model)

    input_shape = (1, 64, 32, 32)
    emb_shape = (1, 64)

    verify_module(
        mod,
        (input_shape, emb_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            intermediates=True,
            verify_last=False
        ),
        uniform_inputs=True,
    )


def test_tvm_downsample_resblock(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    model = init_resblock(32, 32, time_embed_dim=64, downsample=True)
    
    mod = PyTorchModule("ResBlockDownsample", model)

    input_shape = (1, 32, 32, 32)
    emb_shape = (1, 64)

    verify_module(
        mod,
        (input_shape, emb_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            intermediates=True,
            verify_last=False
        ),
        uniform_inputs=True,
    )
