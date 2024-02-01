# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import torch.nn as nn

import pybuda
from pybuda import (
    TTDevice,
    BackendType,
    pybuda_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
    CompileDepth,
)
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda

from test.legacy_tests.clip_guided_diffusion.unet.pytorch_unet import TimestepEmbedSequential, AttentionBlock, ResBlock, QKVAttentionLegacy, timestep_embedding
from test.legacy_tests.clip_guided_diffusion.unet.test_attention_block import init_attention_block


def default_res_block_config():
    config = dict(
        dropout=0.0,
        dims=2,
        emb_channels=1024,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )
    return config


def default_attention_block_config():
    config = dict(num_heads=4, num_head_channels=64)
    return config


def init_res_and_attention_blocks(
    first_res_block_config,
    add_attention_block=False,
    attention_config=None,
    add_second_res_block=False,
    second_res_block_config=None,
):
    layers = []
    assert (
        add_attention_block or add_second_res_block
    ), "Expected to add either an AttentionBlock or a second ResBlock. Tests for standalone ResBlock should not be added here!"

    layers = [ResBlock(**first_res_block_config)]
    if add_attention_block:
        assert attention_config is not None
        layers.append(AttentionBlock(**attention_config))
    if add_second_res_block:
        assert second_res_block_config is not None
        layers.append(ResBlock(**second_res_block_config))

    return TimestepEmbedSequential(*layers)


def two_block_res_model(ch, out_ch, down, up):
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    attention_block_config = default_attention_block_config()
    attention_block_config.update(dict(channels=ch))
    second_res_block_config = default_res_block_config()
    second_res_block_config.update(dict(channels=ch, out_channels=out_ch, down=down, up=up))
    return init_res_and_attention_blocks(
        first_res_block_config,
        add_attention_block=True,
        attention_config=attention_block_config,
        add_second_res_block=True,
        second_res_block_config=second_res_block_config,
    )


# TODO: [A.Sh] Why is this test slow? # Verification is taking a long time
def test_tvm_unet_resblock_attention_block_upsample_resblock(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    
    ch = 1536
    out_ch = 1024
    model = two_block_res_model(ch=ch, out_ch=out_ch, down=False, up=True)
    mod = PyTorchModule("attention_block_upsample", model)
    act1 = torch.randn(1, 1536, 32, 32)
    torch_emb = torch.randn(1, 1024)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "attetion_block_upsample", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute, 
            compile_depth=CompileDepth.BALANCER_PASS), 
            verify_cfg=VerifyConfig(intermediates=True),)

    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)


# TODO: WHAT IS GOING ON?
def test_tvm_adaptive_avg_pool(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference
    class AdaptiveAveragePool(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, target_output=224):
            return torch.nn.functional.adaptive_avg_pool2d(x1, target_output)

    model = AdaptiveAveragePool()
    mod = PyTorchModule("adaptive_ave_pool", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    # TODO: add another test with a simpler shape
    shape = (1, 3, 239, 239)
    act1 = torch.rand(*shape)
    ret = pybuda_compile(tt0, 
    "adaptive_ave_pool", 
    act1, 
    compiler_cfg=CompilerConfig(enable_training=training, 
    enable_recompute=recompute, compile_depth=CompileDepth.POST_INITIAL_GRAPH_PASS), 
    verify_cfg=VerifyConfig(intermediates=True),)

    evaluate_framework_vs_pybuda(model, ret, act1)


def test_tvm_unet_resblock_downsample_resblock(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    ch = 512
    out_ch = 512
    
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    second_res_block_config = default_res_block_config()
    second_res_block_config.update(dict(channels=ch, out_channels=out_ch, down=True))
    
    model = init_res_and_attention_blocks(
        first_res_block_config,
        add_second_res_block=True,
        second_res_block_config=second_res_block_config,
    )
    mod = PyTorchModule("resblock_downsample_resblock", model)

    act1 = torch.randn(1, 512, 64, 64)
    torch_emb = torch.randn(1, 1024)
    
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "resblockdownsampleresblock", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True))
    
    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)


def test_tvm_unet_resblock_upsample_resblock(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    ch = 768
    out_ch = 512
    
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    second_res_block_config = default_res_block_config()
    second_res_block_config.update(dict(channels=out_ch, out_channels=out_ch, up=True))
    model = init_res_and_attention_blocks(
        first_res_block_config,
        add_second_res_block=True,
        second_res_block_config=second_res_block_config,
    )

    mod = PyTorchModule("resblock_upsample_resblock", model)

    act1 = torch.randn(1, 768, 64, 64)
    torch_emb = torch.randn(1, 1024)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "resblockupsampleresblock", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True))
    
    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)
    

def test_tvm_unet_resblock_attention_block(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    ch = 512
    out_ch = 512
    
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    attention_block_config = default_attention_block_config()
    attention_block_config.update(dict(channels=out_ch))
    model = init_res_and_attention_blocks(
        first_res_block_config,
        add_attention_block=True,
        attention_config=attention_block_config,
    )

    mod = PyTorchModule("attention_block", model)

    act1 = torch.randn(1, 512, 32, 32)
    torch_emb = torch.randn(1, 1024)
    

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "attetion_block", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True),)
    
    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)



def test_tvm_unet_resblock_attention_block_resblock(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    ch = 1024
    out_ch = 1024
    model = two_block_res_model(ch=ch, out_ch=out_ch, down=False, up=False)
    mod = PyTorchModule("attention_block_resblock", model)

    act1 = torch.randn(1, 1024, 8, 8)
    torch_emb = torch.randn(1, 1024)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "attetion_block_resblock", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True),)

    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)


def test_tvm_unet_resblock_attention_block_downsample_resblock(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    ch = 512
    out_ch = 512
    model = two_block_res_model(ch=ch, out_ch=out_ch, down=True, up=False)
    mod = PyTorchModule("attention_block_downsample", model)

    act1 = torch.randn(1, 512, 64, 64)
    torch_emb = torch.randn(1, 1024)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "attetion_block_downsample", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True),)
    
    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)


@pytest.mark.parametrize("shape", ((1, 256, 256, 256), (1, 256, 128, 128)),)
@pytest.mark.parametrize("num_groups", (32, ),) # 1 group works with FULL
def test_tvm_group_norm(shape, num_groups, training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class GroupNorm(nn.Module):
        
        def __init__(self, num_groups, num_channels):
            super().__init__()
            self.gn = torch.nn.GroupNorm(num_groups, num_channels)
            
        def forward(self, x, ):
            return self.gn(x)
            
            
    shape = shape
    num_groups = num_groups
    
    num_channels = shape[1]
    
    act1 = torch.rand(*shape)

    model = GroupNorm(num_groups, num_channels)
    mod = PyTorchModule("group_norm", model)
    
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(tt0, "groupnorm", act1, 
        compiler_cfg=CompilerConfig(
        enable_training=training, 
        enable_recompute=recompute, 
        compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER),
        verify_cfg=VerifyConfig(intermediates=True),)
    
    evaluate_framework_vs_pybuda(model, ret, act1)


def test_tvm_unet_upsample(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class Upsample(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x1, ):
            return torch.nn.functional.interpolate(x1, scale_factor=2, mode="nearest")
            
    model = Upsample()
    mod = PyTorchModule("upsample", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 4, 8, 12)
    # [1 x 1024 x 8 x 8]
    # [1 x 1024 x 16 x 16]
    # [1 x 1024 x 32 x 32]
    # [1 x 512 x 64 x 64]
    # [1 x 256 x 128 x 128]
    act1 = torch.rand(*shape)
    ret = pybuda_compile(
        tt0, 
        "upsample", 
        act1, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute, 
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True),)

    evaluate_framework_vs_pybuda(model, ret, act1)


def test_tvm_avg_pool(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class AveragePool(nn.Module):
        def __init__(self):
            super().__init__()            

        def forward(self, x1):
            return torch.nn.functional.avg_pool2d(x1, kernel_size=2, stride=2)
            
    model = AveragePool()
    mod = PyTorchModule("ave_pool", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 1, 64, 64)
    stride=2
    kernel_size=2
    act1 = torch.rand(*shape)

    ret = pybuda_compile(
        tt0, 
        "ave_pool", 
        act1, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True),)
    
    evaluate_framework_vs_pybuda(model, ret, act1)


def test_tvm_qkv_attention(training=False, recompute=False):

    if not training and recompute:
        pytest.skip()

    num_heads = 4
    model = QKVAttentionLegacy(num_heads)
    
    acts = torch.randn(1, 1536, 1024)

    mod = PyTorchModule("qkv_attn_reshape", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "qkv_attn_reshape",
        acts,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.FULL
        ),
        verify_cfg=VerifyConfig(intermediates=True),
    )
    
    evaluate_framework_vs_pybuda(model, ret, acts)


def test_tvm_attention_block(training=False, recompute=False):

    if not training and recompute:
        pytest.skip()

    channels = 512
    if channels == 512:
        act1 = torch.randn(1, 512, 32, 32)
    else:
        assert channels == 1024
        act1 = torch.randn(1, 1024, 8, 8)
    

    model = init_attention_block(channels)
    mod = PyTorchModule("attn_block", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "attn_block",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER
        ),
        verify_cfg=VerifyConfig(intermediates=True),
    )
    evaluate_framework_vs_pybuda(model, ret, act1)
    

def test_tvm_timestep_embed_sequential(training=False, recompute=False):
    pytest.skip()
    # we are running time embeddings on CPU
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    ch = 512
    out_ch = 512
    
    first_res_block_config = default_res_block_config()
    first_res_block_config.update(dict(channels=ch, out_channels=out_ch))
    model = TimestepEmbedSequential(ResBlock(**first_res_block_config))

    mod = PyTorchModule("TimestepEmbedSequential", model)

    act1 = torch.randn(1, 512, 64, 64)
    torch_emb = torch.randn(1, 1024)
    

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "TimestepEmbedSequential", 
        act1, 
        torch_emb, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True))
    
    evaluate_framework_vs_pybuda(model, ret, act1, torch_emb)
    

def test_tvm_timestep_embedding(training=False, recompute=False):
    """ 
        embedding will be executed on CPU
    """
    pytest.skip()
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference
    
    class TimestepEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_channels = 256

        def forward(self, x1):
            return timestep_embedding(x1, self.model_channels)

    model = TimestepEmbedding()
    mod = PyTorchModule("TimestepEmbedding", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = torch.randint(0, 1, size=(1,))

    ret = pybuda_compile(
        tt0, 
        "TimestepEmbedding", 
        act1, 
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute, 
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=VerifyConfig(intermediates=True),)
    
    evaluate_framework_vs_pybuda(model, ret, act1)