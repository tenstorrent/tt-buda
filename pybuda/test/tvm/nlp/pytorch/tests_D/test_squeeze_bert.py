# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest

import torch
import torch.nn as nn
# from transformers.models.squeezebert import SqueezeBertEncoder
from transformers import SqueezeBertModel, SqueezeBertConfig

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


def test_tvm_SqueezeBertEncoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER 

    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    input_shape = (1, 32, 768)

    config = SqueezeBertConfig()
    config.num_hidden_layers = 1
    model = SqueezeBertModel(config)

    mod = PyTorchModule("SqueezeBertEncoder", model.encoder)

    attention_mask = torch.ones(input_shape[0:2])
    extended_attn_mask = model.get_extended_attention_mask(attention_mask, input_shape[0:2], "cpu")
    verify_module(
        mod,
        (input_shape, extended_attn_mask.shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )



def test_tvm_SqueezeBertPooler(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER 

    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    config = SqueezeBertConfig()

    model = SqueezeBertModel(config)

    mod = PyTorchModule("SqueezeBertPooler", model.pooler)

    input_shape = (1, 8, 768)

    verify_module(
        mod,
        (input_shape, ),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
