# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest
from collections import OrderedDict

import torch
from torch import nn
from loguru import logger

from transformers import BertModel, BertConfig, BertForPreTraining, TFBertMainLayer, TFBertForQuestionAnswering

import pybuda
from pybuda import (
    PyTorchModule,
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.tensor import to_pt_tensors
from pybuda.op.eval import compare_tensor_to_golden
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.backend.models.test_bert import get_relaxed_atol_pcc


def test_tvm_passthrough(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"} 
    compiler_cfg.retain_tvm_python_files = True

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 32)
            self.lin = nn.Linear(32, 32)

        def forward(self, x):
            output = (x, )
            x = self.emb(x)
            output = (x, output)
            x = self.lin(x)
            output = (x, output)
            x = x*self.emb.weight
            return (x, output)

    mod = PyTorchModule("passthrough", Model())
    input_shape = (1, 32)
    inp = [torch.randint(0, 32, input_shape)]
    verify_module(
        mod,
        (input_shape,),
        inputs=[inp],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )
