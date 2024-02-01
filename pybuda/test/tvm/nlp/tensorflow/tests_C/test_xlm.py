# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest

import tensorflow as tf
from transformers.models.xlm import XLMConfig
from transformers.models.xlm.modeling_tf_xlm import TFXLMMultiHeadAttention, TFXLMTransformerFFN

import math
import itertools
from pybuda import (
    TFModule,
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


def test_tvm_xlm_attention_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    class TFXLM_MHA(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.config = XLMConfig()
            self.layer = TFXLMMultiHeadAttention(self.config.n_heads, self.config.emb_dim, self.config)

        def call(self, hidden_states, mask):
            return self.layer(hidden_states, mask, None, None, None, False, False)

    model = TFXLM_MHA()

    mod = TFModule("XLM_attention_tf", model)

    input_shape = (1, 16, 2048)
    mask_shape = (1, 16)
    verify_module(
        mod,
        (input_shape, mask_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        uniform_inputs=True,
    )

def test_tvm_xlm_FFN_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    class TFXLM_FFN(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.config = XLMConfig()
            self.layer = TFXLMTransformerFFN(
                self.config.emb_dim,
                self.config.emb_dim * 4, 
                self.config.emb_dim,
                self.config)

        def call(self, hidden_states):
            return self.layer(hidden_states,)

    model = TFXLM_FFN()

    mod = TFModule("XLM_ffn_tf", model)

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
