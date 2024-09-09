# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from codeop import Compile
import sys
from typing import Optional, Tuple
import pytest

import torch
import tensorflow as tf
import numpy as np

from transformers import GPTJConfig, shape_list
from transformers.models.gptj.modeling_tf_gptj import TFGPTJModel, TFGPTJBlock, TFGPTJAttention, TFGPTJMLP
from transformers.modeling_tf_utils import get_initializer

from pybuda import (
    TFModule,
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

def fixed_pos_embedding(x: tf.Tensor, seq_dim: int = 1, seq_len: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    dim = shape_list(x)[-1]
    if seq_len is None:
        seq_len = shape_list(x)[seq_dim]
    inv_freq = tf.cast(1.0 / (10000 ** (tf.range(0, dim, 2) / dim)), tf.float32)
    seq_len_range = tf.cast(tf.range(seq_len), tf.float32)
    sinusoid_inp = tf.cast(tf.einsum("i , j -> i j", seq_len_range, inv_freq), tf.float32)
    return tf.cast(tf.sin(sinusoid_inp), dtype=x.dtype), tf.cast(tf.cos(sinusoid_inp), dtype=x.dtype)



def apply_rotary_pos_emb(x: tf.Tensor, sincos: tf.Tensor, offset: int = 0) -> tf.Tensor:
    sin_pos, cos_pos = sincos
    sin_pos = tf.repeat(sin_pos[None, offset : shape_list(x)[1] + offset, None, :], 2, 3)
    cos_pos = tf.repeat(cos_pos[None, offset : shape_list(x)[1] + offset, None, :], 2, 3)
    return (x * cos_pos) + (rotate_every_two(x) * sin_pos)





@pytest.mark.skip(reason="Tested with fallback")
def test_gptj_block(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()

    config = GPTJConfig(n_layer=1)  # for faster loading
    config.rotary_dim = 64
    model = TFGPTJBlock(config)

    mod = TFModule("gptj_block_tf", model)

    input_shape = (1, 128, 4096)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_gptj_fallback(test_kind, test_device):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()
    if test_kind.is_training(): # only run recompute test in post-commit
        pytest.skip()

    if test_device.arch == BackendDevice.Wormhole_B0:
        pytest.skip() # see tenstorrent/pybuda#969
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True

    #Fusing disabled due to tenstorrent/pybuda#789
    compiler_cfg.enable_auto_fusing=False

    configuration = GPTJConfig(n_layer=1)
    configuration.rotary_dim = 64
    configuration.use_cache = False
    model = TFGPTJModel(configuration)

    mod = TFModule("gptj_tf_fallback", model)

    input_shape = (1, 128)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        input_params=[
            {"requires_grad": False, "data_format": tf.int32}, 
        ],
    )

