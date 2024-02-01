# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest
import tensorflow as tf
from transformers import XLNetConfig
from transformers.models.xlnet.modeling_tf_xlnet import TFXLNetLayer,TFXLNetMainLayer

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
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_tvm_xlm_attention_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    class TFXLNet_Layer(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.layer = TFXLNetLayer(config)

        def call(self, hidden_states,pos_emb):
            # Pybuda -> TVM compile removes batch dim.
            hidden_states = tf.transpose(hidden_states, perm=[1, 0, 2])
            pos_emb = tf.transpose(pos_emb, perm=[1, 0, 2])

            return self.layer(hidden_states, None, None, None, pos_emb, None, None, None, None, False, False)

    config = XLNetConfig()
    submodel = TFXLNet_Layer(config)

    mod = TFModule("XLM_attention_tf", submodel)

    input_shape = (1, 16, 1024)
    pos_emb = (1, 32, 1024)

    verify_module(
        mod,
        (input_shape, pos_emb, ),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        uniform_inputs=True,
    )