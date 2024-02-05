# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import numpy as np
import tensorflow as tf
from transformers import GPT2Config
from transformers.models.gpt2.modeling_tf_gpt2 import TFBlock, TFGPT2Model

from pybuda import (
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.backend.models.test_bert import get_relaxed_atol_pcc

@pytest.mark.skip(reason="Tested with fallback")
def test_gpt2_block_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    class TF_GPT2Block(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFBlock(config=config)

        def call(self, hidden_states):
            return self.layer(hidden_states, None, None, None, None, None, None, None)

    config = GPT2Config()
    model = TF_GPT2Block(config=config)
    mod = TFModule("GPT2Block", model)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn/c_attn/weight:0", "attn/c_attn/bias:0"}


    input_shape = (1, 64, 768)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"c_attn/bias"},
        )
    )

def test_tvm_gpt2_fallback(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()
        #TODO: Fix tvm .14 regressions: tenstorrent/pybuda#2099

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn/c_attn/weight", "attn/c_attn/bias"} 

    input_shape = (1, 768)
   
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 1
    config.use_cache = False
    model = TFGPT2Model(config)

    mod = TFModule("gpt2", model)
    
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"c_attn/bias"},
            relative_atol=relative_atol,
            pcc=pcc,
        ),
        input_params=[{"requires_grad": False, "data_format": tf.int32}],
    )
