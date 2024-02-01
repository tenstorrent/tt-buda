# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os
import pytest
from collections import OrderedDict

import torch
from torch import nn
from test.backend.models.test_bert import get_relaxed_atol_pcc
from pybuda.tensor import to_pt_tensors

from transformers import BertModel, BertConfig, BertForPreTraining
import pybuda
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.op.eval import compare_tensor_to_golden
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


@pytest.mark.parametrize("size", ["tiny", "base", "large"])
def test_bert_encoder(test_kind, test_device, size):
    if size == "tiny":
        model_name = "prajjwal1/bert-tiny"
        seq_len = 128
    elif size == "base":
        model_name = "bert-base-uncased"
        seq_len = 128
    elif size == "large":
        model_name = "bert-large-uncased"
        seq_len = 384

    pytest.skip("Full model passes inference and training")
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    config = download_model(BertConfig.from_pretrained, model_name)
    input_shape = (1, seq_len, config.hidden_size)
    model = download_model(BertModel.from_pretrained, model_name, torchscript=True)

    submodel = model.encoder
    mod = PyTorchModule("bert_encoder", submodel)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"layer.0.attention.self.key.bias", "layer.1.attention.self.key.bias"},
        ),
        input_params=[{"requires_grad": False}],
    )
    # evaluate_framework_vs_pybuda(submodel, ret, hidden_states)

def test_pt_pretrain_heads(test_device):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    test_device.devtype = BackendType.NoBackend
    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny", torchscript=True)
    bert = BertForPreTraining(config)
    submodel = bert.cls
    mod = PyTorchModule("ptheads", submodel)
    verify_module(
        mod,
        ((1, 128, 128), (1, 128)),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        )
    )

def test_bert_pt_fallback(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()
        
    input_shape = (1, 128)
    model = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", add_pooling_layer=False)

    mod = PyTorchModule("bert", model)

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.retain_tvm_python_files = True

    pcc = 0.9 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=pcc,
            waive_gradient_errors={"layer.0.attention.self.key.bias", "layer.1.attention.self.key.bias"},
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )


def test_bert_embeddings_fallback(test_kind, test_device):
    pytest.skip("Full model passes inference and training")
    class EmbModel(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb
            self.linear = nn.Linear(128, 32)

        def forward(self, input):
            embs = self.emb(input)
            lin = self.linear(embs)
            return lin


    compiler_cfg = _get_global_compiler_config() 
    input_shape = (1, 32)

    bert = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", torchscript=True)    
    mod = PyTorchModule("bert_embedding", EmbModel(bert.embeddings))

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )

def test_bert_direct_fallback(test_kind, test_device):
    pytest.skip("Full model passes inference and training")

    compiler_cfg = _get_global_compiler_config() 

    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny")
    config.num_hidden_layers = 2
    model = BertModel(config, add_pooling_layer=False)

    mod = PyTorchModule("bert", model)
    tt1 = pybuda.TTDevice("tt1",
            devtype=test_device.devtype, arch=test_device.arch, module=mod)
    input_shape = (1, 128)
    input_ids = torch.randint(high=25000, size=input_shape)
    attention_mask = torch.ones(input_shape)

    tt1.push_to_inputs(input_ids, attention_mask)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(relative_atol=0.3), _sequential=True)
    output = to_pt_tensors(output_q.get())[0]

    pt_output = model(input_ids, attention_mask)[0]

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, "tiny", 1)
    compare_tensor_to_golden("bert_out", pt_output, output, pcc=pcc, relative_atol=relative_atol)
