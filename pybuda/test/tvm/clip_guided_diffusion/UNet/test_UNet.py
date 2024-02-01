# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
import torch.nn as nn

import pybuda

from pybuda import (
    TTDevice,
    CPUDevice, 
    BackendType,
    pybuda_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
)
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.verify import verify_module_pipeline

from test.legacy_tests.clip_guided_diffusion.unet.pytorch_unet import UNetModel, create_UNet, timestep_embedding


@pytest.mark.parametrize("depth", (CompileDepth.POST_INITIAL_GRAPH_PASS, ), )
def test_tvm_unet(depth, training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    learn_sigma = True
    out_channels = 3 if not learn_sigma else 6
    class_cond = False
    num_classes = 1000 if class_cond else None
    config = {
        'image_size': 256,
        'in_channels': 3,
        'model_channels': 256,
        'out_channels': out_channels, 
        'num_res_blocks': 2,
        'attention_resolutions': (8, 16, 32),
        'dropout': 0.0,
        'channel_mult': (1, 1, 2, 2, 4, 4),
        'num_classes': num_classes,
        'use_checkpoint': False,
        'use_fp16': False,
        'num_heads': 4,
        'num_head_channels': 64,
        'num_heads_upsample': 1,
        'use_scale_shift_norm': True,
        'resblock_updown': True,
        'use_new_attention_order': False,
    }
            
    model = UNetModel(**config)
    mod = PyTorchModule("UNet", model)

    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    shape = (1, 3, 256, 256)
    act1 = torch.rand(*shape)
    timesteps =  torch.randint(0, 1, size=(1,)).float()

    ret = pybuda_compile(
        tt0, 
        "UNet", 
        act1, 
        timesteps,
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute, 
            compile_depth=depth), 
            verify_cfg=VerifyConfig(intermediates=True),)
    
    evaluate_framework_vs_pybuda(model, ret, act1, timesteps)


def test_tvm_unet_time_embed(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()
    
    class TimeEmb(nn.Module):
        
        def __init__(self, model_channels, time_embed_dim):
            super().__init__()
            self.time_emb = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        
        def forward(self, time_embedded_steps):
            return self.time_emb(time_embedded_steps)
        
    UNet_embeddings, UNet_no_emb, UNetTorchModel, UNet_config = create_UNet()
    
    verify_cfg = VerifyConfig(intermediates=False)
    
    act1_shape = (1, 3, 256, 256)
    act1 = torch.rand(*act1_shape)
    timesteps_shape = (1, )
    timesteps =  torch.randint(0, 1, size=timesteps_shape, requires_grad=False).float()
    
    act1, embedded_res = UNet_embeddings(timesteps, act1)
    model = TimeEmb(UNet_config['model_channels'], UNet_config['model_channels'] * 4)
    
    mod = PyTorchModule('time_emb',model)
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    
    ret = pybuda_compile(
        tt0, 
        "time_emb", 
        embedded_res,
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute, 
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER), 
            verify_cfg=verify_cfg,)

    evaluate_framework_vs_pybuda(model, ret, embedded_res)


def test_tvm_unet_emb_precomp(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    verify_cfg = VerifyConfig(intermediates=False, verify_last=False)

    UNet_embeddings, UNet_no_emb, UNetTorchModel, UNet_config = create_UNet()
    act1_shape = (1, 3, 256, 256)
    act1 = torch.rand(*act1_shape)
    timesteps_shape = (1, )
    timesteps =  torch.randint(0, 1, size=timesteps_shape, requires_grad=False).float()
    
    act1, embedded_res = UNet_embeddings(timesteps, act1)

    UNet_mod = PyTorchModule('UNet_model',UNet_no_emb)
    sgd_optimizer = pybuda.optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(UNet_mod)

    ret = pybuda_compile(
        tt0, 
        "UNet", 
        act1, 
        embedded_res,
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute, 
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER, 
            compile_tvm_to_python=False), 
            verify_cfg=verify_cfg,)
    evaluate_framework_vs_pybuda(UNet_no_emb, ret, act1, embedded_res)


def test_tvm_splitted_unet(training=False, recompute=False):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    verify_cfg = VerifyConfig(intermediates=True)

    UNet_embeddings, UNet_model, UNetTorchModel = create_UNet()
    act1_shape = (1, 3, 256, 256)
    act1 = torch.rand(*act1_shape)
    timesteps_shape = (1, )
    timesteps =  torch.randint(0, 1, size=(1,), requires_grad=False).float()
    
    device_types=["CPUDevice", "TTDevice"]
    
    embedding_module = PyTorchModule('UNet_time_embedding', UNet_embeddings)
    

    UNet_mod = PyTorchModule('UNet_model',UNet_model)


    verify_module_pipeline([embedding_module, UNet_mod], 
                    [timesteps_shape, act1_shape],
                    VerifyConfig(training=training, accumulation_steps=1, intermediates=False), 
                    input_params = [{"requires_grad": False, "data_format": torch.float32}, {"requires_grad": False, "data_format": torch.float32}],
                    device_types=["CPUDevice", "TTDevice"])

