# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import configparser
from distutils.command.config import config
import pybuda
import pytest

import torch
from transformers import AlbertConfig, AlbertModel

from pybuda import (
    PyTorchModule,
    TTDevice,
    CPUDevice,
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

from pybuda.op.eval.common import compare_tensor_to_golden
from test.utils import download_model


class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.embeddings

    def forward(self, input_ids, extended_attention_mask):
        return self.embeddings(input_ids), extended_attention_mask

class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.pooler = model.pooler
        self.pooler_activation = model.pooler_activation

    def forward(self, embedding_output, extended_attention_mask):
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
        )
        return encoder_outputs


@pytest.mark.parametrize("add_pooling_layer", [True, False], ids=["pooling", "no_pooling"])
@pytest.mark.parametrize("version", ['v1', 'v2'], )
def test_albert_pipeline(test_device, version, add_pooling_layer):
    if add_pooling_layer:
        pytest.skip("Pooling not supported in backend, will result in unsupported sparse_matmul")
    
    config = download_model(AlbertConfig.from_pretrained, f"albert-base-{version}", torchscript=True)
    model = AlbertModel(config, add_pooling_layer=add_pooling_layer)
    model.eval()
    
    albert_embeddings = EmbWrapper(model)
    albert_encoder = EncoderWrapper(model)

    relative_atol = 0.3 if test_device.is_silicon() else 0.1

    cpu0 = CPUDevice("cpu0", module=PyTorchModule("albert_embeddings", albert_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("albert_encoder", albert_encoder))

    seq_len = 128
    input_ids = torch.randint(config.vocab_size, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=model.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    cpu0.push_to_inputs(input_ids, extended_attention_mask)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(relative_atol=relative_atol))
    outputs = output_q.get()

    torch_outputs = model(input_ids, attention_mask=attention_mask)
    assert compare_tensor_to_golden("albert", torch_outputs[0], outputs[0].value(), is_buda=True, relative_atol=relative_atol)


def test_albert_v1(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    
    input_shape = (1, 768, 768)
    
    model = download_model(AlbertModel.from_pretrained, "albert-base-v1", torchscript=True)

    submodel = model.encoder.albert_layer_groups[0].albert_layers[0].attention
    mod = PyTorchModule(
        "albert_attention_pytorch",
        submodel,
    )
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"key.bias"},
        )
    )
    # evaluate_framework_vs_pybuda(submodel, res, hidden_states)

def test_albert_v2(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 768, 768)

    model = download_model(AlbertModel.from_pretrained, "albert-base-v2", torchscript=True)

    submodel = model.encoder.albert_layer_groups[0].albert_layers[0].attention
    mod = PyTorchModule(
        "albert_attention_pytorch",
        submodel,
    )
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"key.bias"},
        )
    )
    # evaluate_framework_vs_pybuda(submodel, res, hidden_states)
