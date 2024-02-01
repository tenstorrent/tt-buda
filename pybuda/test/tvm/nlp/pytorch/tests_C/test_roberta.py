# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
from pybuda.config import CompileDepth
from pybuda.cpudevice import CPUDevice
from pybuda.verify.cpueval import TrainingEvalData
import pytest
from loguru import logger


import torch
from transformers import RobertaModel, RobertaConfig

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
from test.backend.models.test_bert import get_relaxed_atol_pcc
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model

class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.embeddings

    def forward(
        self,
        input_ids,
        extended_attention_mask,
    ) -> torch.Tensor:
        return self.model(input_ids), extended_attention_mask

class RobertaEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.pooler = model.pooler

    def forward(self, embedding_output, extended_attention_mask):
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        return encoder_outputs

def test_roberta_pipeline(test_kind, test_device):
    pytest.skip("Full model passes inference and training")
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        pass#compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        pass# compiler_cfg.compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS

    config = RobertaConfig()
    model = RobertaModel(config, add_pooling_layer=False)
    model.eval()

    roberta_embeddings = EmbWrapper(model)
    roberta_encoder = RobertaEncoder(model)


    cpu0 = CPUDevice("cpu0", module=PyTorchModule("roberta_embeddings", roberta_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("roberta_encoder_stack", roberta_encoder))

    seq_len = 128
    input_ids = torch.randint(config.vocab_size, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))
    extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size())
    cpu0.push_to_inputs(input_ids, extended_attention_mask)
    # tt1.push_to_inputs(input_ids)
    output_q = pybuda.run_inference()
    outputs = output_q.get()

    torch_outputs = model(input_ids)
    assert compare_tensor_to_golden("roberta", torch_outputs[0], outputs[0].value(), is_buda=True)


def test_roberta_encoder(test_kind, test_device):
    pytest.skip("Full model passes training and inference")
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    input_shape = (1, 256, 256)
    roberta_model = download_model(RobertaModel.from_pretrained, "arampacha/roberta-tiny", torchscript=True)
    model = roberta_model.encoder

    hidden_states = torch.rand(*input_shape)

    mod = PyTorchModule("roberta_encoder_pytorch", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"layer.0.attention.self.key.bias", "layer.1.attention.self.key.bias", 
            "layer.2.attention.self.key.bias", "layer.3.attention.self.key.bias"} # small numbers
        )
    )

def test_roberta_full(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 128)
    model = download_model(RobertaModel.from_pretrained, "arampacha/roberta-tiny", torchscript=True)
    model.pooler = None
    model.return_dict = False

    class RobertaWrapper(torch.nn.Module):
        def __init__(self, roberta):
            super().__init__()
            self.roberta = roberta

        def forward(self, x):
            out = self.roberta(x)
            
            return [output for output in out if output is not None]

    input_ids = [torch.randint(0, input_shape[-1], input_shape)]

    mod = PyTorchModule("roberta_encoder_pytorch", RobertaWrapper(model))

    verify_module(
        mod,
        (input_shape,),
        inputs=[input_ids],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.97,
            waive_gradient_errors={"layer.0.attention.self.key.bias", "layer.1.attention.self.key.bias", 
             "layer.2.attention.self.key.bias", "layer.3.attention.self.key.bias"} # small numbers
        )
    )

