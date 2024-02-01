# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os
import torch
import pytest
from transformers import DistilBertModel

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


def test_distilbert_pt(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Transformer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.attn_mask = torch.ones((1, 128))
            self.module = module

        def forward(self, input_act):
            return self.module(input_act, self.attn_mask)

    framework_module = download_model(
        DistilBertModel.from_pretrained,
        "distilbert-base-cased-distilled-squad"
    )
    framework_module = Transformer(framework_module)
    pybuda_module = PyTorchModule("distilbert_pt", framework_module)

    # Input shapes
    input_act_shape = (1, 128)

    # Sanity check
    # act = torch.randint(0, 25000, input_act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
            waive_gradient_errors={"attention.k_lin.bias"},
        ),
        input_params=[{"data_format": torch.int}],
    )


def test_distilbert_layer_pt(test_kind, test_device):
    pytest.skip("Covered in full DistilBert test")
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Transformer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attn_mask = torch.ones((1, 128))

        def forward(self, input_act):
            return self.module.transformer.layer[0](input_act, self.attn_mask)

    framework_module = download_model(
        DistilBertModel.from_pretrained,
        "distilbert-base-cased-distilled-squad"
    )
    framework_module = Transformer(framework_module)
    pybuda_module = PyTorchModule("distilbert_layer_pt", framework_module)

    # Input shapes
    input_act_shape = (1, 128, 768)

    # Sanity check
    # act = torch.rand(input_act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
            verify_all=True,
        ),
    )


def test_distilbert_layer_mha_pt(test_kind, test_device):
    pytest.skip("Covered in full DistilBert test")
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Transformer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attn_mask = torch.ones((1, 128))

        def forward(self, q_act, k_act, v_act):
            return self.module.transformer.layer[0].attention(
                q_act, k_act, v_act, self.attn_mask
            )

    framework_module = download_model(
        DistilBertModel.from_pretrained,
        "distilbert-base-cased-distilled-squad"
    )
    framework_module = Transformer(framework_module)
    pybuda_module = PyTorchModule("distilbert_layer_mha_pt", framework_module)

    # Input shapes
    inp_shape = (1, 128, 768)

    # Sanity check
    q_act = torch.rand(inp_shape)
    k_act = torch.rand(inp_shape)
    v_act = torch.rand(inp_shape)
    out = framework_module(q_act, k_act, v_act)

    verify_module(
        pybuda_module,
        (inp_shape, inp_shape, inp_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
            verify_all=True,
            waive_gradient_errors={"attention.k_lin.bias"}
        ),
    )


def test_distilbert_layer_with_embeddings_pt(test_kind, test_device):
    pytest.skip("Covered in full DistilBert test")
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # Test only inference
    if test_kind.is_training():
        pytest.skip()

    os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"

    class Transformer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.attn_mask = torch.ones((1, 128))
            self.module = module

        def forward(self, input_act):
            # return self.module(input_act, self.attn_mask)
            emb_out = self.module.embeddings(input_act)
            return self.module.transformer.layer[0](emb_out, self.attn_mask)

    framework_module = download_model(
        DistilBertModel.from_pretrained,
        "distilbert-base-cased-distilled-squad"
    )
    framework_module = Transformer(framework_module)
    pybuda_module = PyTorchModule(
        "distilbert_layer_with_embeddings_pt", framework_module
    )

    # Input shapes
    input_act_shape = (1, 128)

    # Sanity check
    # act = torch.randint(0, 25000, input_act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
            # waive_gradient_errors={
            #     "attention/k_lin/bias:0",
            #     "LayerNorm.weight",
            #     "LayerNorm.bias",
            # },
            verify_all=True,
        ),
        input_params=[{"data_format": torch.int}],
    )


def test_distilbert_without_embeddings_pt(test_kind, test_device):
    pytest.skip("Covered in full DistilBert test")
    # Test only inference
    if test_kind.is_training():
        pytest.skip()

    class Transformer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.attn_mask = torch.ones((1, 32))
            self.module = module

        def forward(self, inputs_embeds):
            return self.module(None, self.attn_mask, None, inputs_embeds)

    framework_module = download_model(
        DistilBertModel.from_pretrained,
        "distilbert-base-cased-distilled-squad", torchscript=True
    )
    framework_module = Transformer(framework_module)
    pybuda_module = PyTorchModule("distilbert_without_embeddings_pt", framework_module)

    # Input shapes
    input_emb_shape = (1, 32, 768)

    # Sanity check
    # inputs_embeds = torch.rand(input_emb_shape)
    # out = framework_module(inputs_embeds)

    verify_module(
        pybuda_module,
        (input_emb_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
