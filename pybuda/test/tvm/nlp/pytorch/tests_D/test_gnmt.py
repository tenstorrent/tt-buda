# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest
from typing import Optional, Tuple

import torch
import torch.nn as nn

import numpy as np
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    Tensor,
)

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import pybuda
from test.tvm.nlp.pytorch.gnmt.gnmt import GNMT
from test.tvm.nlp.pytorch.gnmt.encoder import ResidualRecurrentEncoder
from test.tvm.nlp.pytorch.gnmt.decoder import ResidualRecurrentDecoder, RecurrentAttention 
from test.tvm.utils import evaluate_framework_vs_pybuda


def test_gnmt_torch(test_kind, test_device): 
    pytest.skip() # Takes too long on post commit
    
    vocab_size = 32320
    model = GNMT(
            vocab_size=vocab_size,
            num_layers=4,
            hidden_size=1024,
            share_embedding=False,
        )
    mod = PyTorchModule("gnmt_torch", model) 

    #input_shapes = ((64, 32,), (32,), (64, 32,))
    input_shapes = ((1, 32,), (32,), (1, 32,))
    input_activations = [
        torch.randint(vocab_size, size=input_shapes[0]),
        torch.from_numpy(
            np.array([input_shapes[0][0]] * input_shapes[1][0]).astype(np.int64)
        ),
        torch.randint(vocab_size, size=input_shapes[2]),
    ]

    verify_module(
        mod,
        input_shapes,
        inputs=[input_activations],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind, 
        ), 
    ) 


def test_gnmt_torch_encoder(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip() 

    vocab_size = 32320 
    submodule = ResidualRecurrentEncoder(vocab_size=vocab_size,
            num_layers=4,
            hidden_size=1024,)
    mod = PyTorchModule("gnmt_torch_encoder", submodule) 

    input_shapes = ((1, 32,),) 
    input_activations = [torch.randint(vocab_size, size=input_shapes[0]),]  

    verify_module(
        mod,
        input_shapes,
        inputs=[input_activations],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind, 
        ), 
    )
    

def test_gnmt_torch_decoder(test_kind, test_device): 
    pytest.skip("op using w dimension is not supported") 

    class DecoderWrapper(nn.Module):
        def __init__(self, is_training):
            super().__init__()
            self.decoder = ResidualRecurrentDecoder(vocab_size=32320,num_layers=4,hidden_size=1024,)
            self.is_training = is_training

        def forward(self, x1, x2, x3):
            context = (x1, x2, None)
            output, _, _ = self.decoder(x3, context, inference=(not self.is_training)) 
            return output  
 
    mod = PyTorchModule("gnmt_torch_decoder", DecoderWrapper(test_kind.is_training)) 

    input_shapes = ((1, 32, 1024,), (32,), (1, 32,))    
    input_activations = [
        torch.rand(input_shapes[0]),
        torch.from_numpy(
            np.array([input_shapes[0][0]] * input_shapes[1][0]).astype(np.int64)
        ),
        torch.randint(32320, size=input_shapes[2]),
    ]

    verify_module(
        mod,
        input_shapes,
        inputs=[input_activations],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind, 
        ), 
    )


def test_gnmt_torch_rnn_attn(test_kind, test_device): 
    pytest.skip("op using w dimension is not supported") 

    class AttnWrapper(nn.Module): 
        def __init__(self): 
            super().__init__()
            hidden_size = 1024
            self.att_rnn = RecurrentAttention(
                hidden_size,
                hidden_size,
                hidden_size,
                num_layers=1,
                batch_first=False,
                dropout=0.2,
            )

        def forward(self, x1, x2, x3):
            out, _, attn, _ = self.att_rnn(x1, None, x2, x3) 
            return out, attn

    mod = PyTorchModule("gnmt_torch_rnn_attn", AttnWrapper()) 
    input_shapes = ((1, 32, 1024), (1, 32, 1024,), (32,))  
    inputs = [
        torch.rand(input_shapes[0]),
        torch.rand(input_shapes[1]),
        torch.from_numpy(
            np.array([input_shapes[0][0]] * input_shapes[2][0]).astype(np.int64)
        ),
    ]

    verify_module(
        mod,
        input_shapes,
        inputs=[inputs], 
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind, 
        ), 
    )
 
