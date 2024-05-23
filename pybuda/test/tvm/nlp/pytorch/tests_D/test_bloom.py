# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendDevice,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
import random

from test.tvm.nlp.pytorch.bloom import GPTModel, Embedding, tinybloom_args, Transformer, init_method

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from transformers import BloomModel
from test.utils import download_model

def test_bloom_model_transposed(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    model = Transformer(transpose_hidden_states=False)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"self_attention.query_key_value.weight", "self_attention.query_key_value.bias"}


    submodel = model
    mod = PyTorchModule("bloom_encoder", submodel)

    input_shape = (16, 512, 128)
    torch_input = torch.rand(input_shape)
    new_res = model(torch_input)

    old_model = Transformer()
    old_model.load_state_dict(model.state_dict())
    old_res = old_model(torch_input)
    assert torch.allclose(old_res, new_res)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_bloom_model(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    model = Transformer()
    compiler_cfg = _get_global_compiler_config()

    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    else:
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS  # Unsupported HW ops

    submodel = model
    mod = PyTorchModule("bloom_encoder", submodel)

    input_shape = (1, 32, 128)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )



def test_bloom_hf(test_kind, test_device):
    if test_kind.is_training():
        # output mismatch
        pytest.skip()

    if test_device.arch == BackendDevice.Wormhole_B0 or test_device.arch == BackendDevice.Blackhole:
        pytest.skip() # see tenstorrent/pybuda#969

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    #Fusing disabled due to tenstorrent/pybuda#789
    compiler_cfg.enable_auto_fusing=False

    class BloomWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids=input_ids)
        
    submodel = download_model(
                BloomModel.from_pretrained, 
                "bigscience/bloom-560m", return_dict=False, 
                use_cache=False)
    submodel.h = submodel.h[0:1]

    mod = PyTorchModule("bloom_hf", BloomWrapper(submodel))

    input_shape = (1, 512)
    input_ids = torch.randint(0, 50257, (1, 512))

    verify_module(
        mod,
        (input_shape,),
        inputs=[(input_ids,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95, # Causal mask generation can't be traced
        ),
        input_params=[
            {"requires_grad": False, "data_format": torch.int}, 
        ],
    )
