# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.verify.backend import verify_module
import pytest

import torch
import torch.nn as nn
from loguru import logger
import pybuda


from pybuda import (
    TTDevice,
    BackendType,
    pybuda_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
)
from pybuda.config import CompileDepth
from pybuda import TTDevice, VerifyConfig, pybuda_compile, Tensor
from test.tvm.utils import evaluate_framework_vs_pybuda

from test.legacy_tests.clip_guided_diffusion.clip.clip_torch import torch_mha, CLIP, QuickGELU, ResidualAttentionBlock, CLIP_N_HEAD, Transformer, VisionTransformer, CLIP_D_MODEL, QuickGELU
from pybuda.config import _get_global_compiler_config


@pytest.mark.parametrize("use_quick_gelu", (True, False), ids=["quick_gelu", "no_quick_gelu"])
def test_tvm_res_attn_block(test_kind, test_device, use_quick_gelu):

    _get_global_compiler_config().compile_depth = CompileDepth.BALANCER_PASS
    model = ResidualAttentionBlock(d_model=CLIP_D_MODEL, n_head=CLIP_N_HEAD, use_quick_gelu=use_quick_gelu)
    
    mod = PyTorchModule("res_attn_block", model)
    verify_module(
        mod,
        ((1, 16, 768),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


def test_tvm_clip_mha_inner_matmuls(test_kind, test_device):

    query_shape = (1, 9216, 1, 64)
    key_shape = (1, 9216, 64, 1)
    value_shape = (1, 9216, 1, 64)
    class MHAInnerMatmuls(nn.Module):
        def __init__(self):
            super().__init__()
            

        def forward(self, query, key, value):
            query_normalized = query / 8
            attention_scores = torch.matmul(query_normalized, key)
            attention_probs = torch.softmax(attention_scores, dim=2)
            return torch.matmul(attention_probs, value)
    
    model = MHAInnerMatmuls()
    mod = PyTorchModule("MHAInnerMatmuls", model)
    verify_module(
        mod,
        (query_shape, key_shape, value_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


def test_tvm_clip_text_embedding(training=False, recompute=False):
    pytest.xfail() # text embedding is going to be executed on CPU
    if not training and recompute:
        pytest.skip()

    class TextEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 49408
            transformer_width = 512
            self.embedding = nn.Embedding(vocab_size, transformer_width)

        def forward(self, x1, ):
            return self.embedding(x1.long())

    
    model = TextEmbedding()
    input_text = torch.rand((16, 1))
    mod = PyTorchModule("text_embedding", model)
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "text_embedding",
        input_text,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.BALANCER_PASS
        ),
        verify_cfg=VerifyConfig(intermediates=True),
    )
    evaluate_framework_vs_pybuda(model, ret, input_text)


def test_tvm_clip_quick_gelu(test_kind, test_device):
    
    model = QuickGELU()
    act1 = torch.rand((1, 1, 32, 32))
    mod = PyTorchModule("quickgelu", model)

    verify_module(
        mod,
        ((1, 1, 32, 32),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


def test_tvm_clip_gelu(test_kind, test_device):

    class Gelu(nn.Module):
        def __init__(self):
            super().__init__()
            

        def forward(self, x1, ):
            return torch.nn.functional.gelu(x1)

    
    model = Gelu()
    act1 = torch.rand((1, 1, 32, 32))
    mod = PyTorchModule("gelu", model)
    verify_module(
        mod,
        ((1, 1, 32, 32),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


def test_tvm_clip_arange(training=False, recompute=False):
    pytest.xfail() # generating tensors is not supported in pybuda/buda
    if not training and recompute:
        pytest.skip()

    class Arange(nn.Module):
        def __init__(self):
            super().__init__()
            

        def forward(self, x1, ):
            return torch.arange(x1.shape[0])

    
    model = Arange()
    act1 = torch.rand((16, 77, 512))
    mod = PyTorchModule("arange", model)
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "arange",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.BALANCER_PASS
        ),
        verify_cfg=VerifyConfig(intermediates=True),
    )
    evaluate_framework_vs_pybuda(model, ret, act1)
    

# @pytest.mark.parametrize("num_blocks", (2, 12), ids=["two", "twelve"])
@pytest.mark.parametrize("num_blocks", (2, ), ids=["two", ])
def test_tvm_clip_multi_resblocks(test_kind, test_device, num_blocks):
    
    _get_global_compiler_config().compile_depth = CompileDepth.BALANCER_PASS
    model  = Transformer(width=CLIP_D_MODEL, layers=num_blocks, heads=CLIP_N_HEAD)
    mod = PyTorchModule("transformer" + str(num_blocks), model)
    
    verify_module(
        mod,
        ((1, 16, 768),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )
    

def test_tvm_clip_rand(training=False, recompute=False):
    pytest.xfail() # we do not support generating tensors in pybuda/buda
    if not training and recompute:
        pytest.skip()
    class Rand(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, ):
            return torch.randn(32, 32)
    
    model = Rand()
    
    mod = PyTorchModule("rand", model)
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "rand",
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.POST_INITIAL_GRAPH_PASS
        ),
        verify_cfg=VerifyConfig(intermediates=True),
    )
    evaluate_framework_vs_pybuda(model, ret, )


def test_tvm_permute(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BALANCER_PASS
    class Permute(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.permute(x, (1, 0, 2))

    model = Permute()

    mod = PyTorchModule("permute", model)
   
    verify_module(
        mod,
        ((1, 3, 224),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


def test_tvm_cat(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BALANCER_PASS
    class Concatenate(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.cat([x1, x2], dim=1)

    model = Concatenate()

    mod = PyTorchModule("permute", model)
    
    verify_module(
        mod,
        ((1, 3, 3, 224), (1, 7, 3, 224)),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )
