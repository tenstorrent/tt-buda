# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import torch.nn as nn
from transformers import FSMTModel

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import CompileDepth, _get_global_compiler_config


class WMT_Encoder_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', tokenizer='moses', bpe='fastbpe')
        model = FSMTModel.from_pretrained("facebook/wmt19-en-de", torchscript=True)
        
        self.mod = model.encoder.layers[0]
        self.encoder_padding_mask = torch.zeros((1, 1)).to(torch.bool)
        self.attn_mask = torch.ones((16,)).to(torch.bool)

    def forward(self, x):
        return self.mod(x, self.encoder_padding_mask, self.attn_mask)[0]


def test_wmt_encoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    submodel = WMT_Encoder_Wrapper()
    mod = PyTorchModule("wmt16_encoder", submodel)

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    else:
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    # out = model(torch.randint(0, 256, (1, 1), dtype=torch.int32))
    out = submodel(torch.rand((1, 1, 1024)))

    verify_module(
        mod,
        ((1, 1, 1024),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


class WMT_Decoder_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # model = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
        model = FSMTModel.from_pretrained("facebook/wmt19-en-de", torchscript=True)

        self.mod = model.decoder.layers[0]

    def forward(self, x):
        encoder_out = torch.ones(x.shape)
        encoder_padding_mask = None
        incremental_state = None
        prev_self_attn_state = None
        prev_attn_state = None
        attn_mask = None
        attn_padding_mask = None
        result = self.mod(
            x,
            encoder_out,
            encoder_padding_mask,
            incremental_state,
            prev_self_attn_state,
            prev_attn_state,
            attn_mask,
            attn_padding_mask,
            True,
        )

        return result[0], result[1]


def test_wmt_decoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    submodel = WMT_Decoder_Wrapper()

    mod = PyTorchModule("wmt16_decoder", submodel)

    if test_kind.is_training():
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    else:
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    verify_module(
        mod,
        ((1, 32, 1024),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
