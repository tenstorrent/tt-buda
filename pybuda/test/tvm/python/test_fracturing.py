# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

import pybuda
from pybuda import (
    Tensor,
    Parameter,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
)
from pybuda.verify import verify_module


class AttentionMatmul(torch.nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh

    def forward(self, q, k):
        '''
        q : 1 x 1 x d
        k:  1 x s x d
        '''
        _, s, d = k.size()
        nh = self.nh
        dh = d // nh

        k = k.view(1, s, nh, dh).permute(0, 2, 3, 1) # nh, dh, s

        q = q.view(1, 1, nh, dh).permute(0, 2, 1, 3) # nh, 32, dh
        attn = torch.matmul(q, k)
        
        return attn

def test_attn_fracture_matmul_heads(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    s = 512
    d = 1024
    nh = 16
    dh = d // nh 
    mod = AttentionMatmul(nh)
    module = pybuda.PyTorchModule('attn_matmul', mod)

    pybuda.set_configuration_options(default_df_override=pybuda.DataFormat.Float32,
        accumulate_df=pybuda.DataFormat.Float32)


    '''
    Fracturing
    '''
    factor = 2
    pybuda.config.insert_fracture_group([("matmul_7", -3, factor),])

    verify_module(module, [(1, 1, d), (1, s, d)],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, chip_ids=[0]),)


class AttentionMatmulLoopback(torch.nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh

    def forward(self, q, k_cache_param, k_new):
        '''
        q : 1 x 1 x d
        k_cache_param: 1 x s x d
        k_new:  1 x 32 x d
        '''
        _, s, d = k_cache_param.size()
        nh = self.nh
        dh = d // nh

        k = k_new.view(1, 32, nh, dh).permute(0, 2, 1, 3) # 1, nh, dh, 32
        k_cache = k_cache_param.view(1, s, nh, dh).permute(0, 2, 1, 3) # 1, nh, dh, s
        k = torch.cat((k_cache, k), dim=-2)

        q = q.view(1, 1, nh, dh).permute(0, 2, 1, 3) # 1, nh, 1, dh
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        return attn, k_new

def test_attn_cache_loopback(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    s = 512
    d = 1024
    nh = 16
    dh = d // nh 
    mod = AttentionMatmulLoopback(nh)
    module = pybuda.PyTorchModule('attn_loopback_fractured', mod)

    df = pybuda.DataFormat.Float16
    pybuda.set_configuration_options(default_df_override=df,
        accumulate_df=df)

    from pybuda.config import  _get_global_compiler_config
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.loopback_outputs = {"k_cache_param": 1}


    # '''
    # Fracturing
    # '''
    factor = 2
    pybuda.config.insert_fracture_group([("concatenate_5", -3, factor)])

    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=module)

    compile_inputs = (torch.randn(1, 1, d), torch.randn(1, s, d), torch.randn(1, 32, d))
    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_inputs,)

    inputs = (torch.randn(1, 1, d), torch.randn(1, 32, d))
    tt0.push_to_inputs(inputs)
    pybuda.run_generate(input_count=1, write_index=0)


class AttentionModuleLoopback(torch.nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh

    def forward(self, q, k_cache_param, k_new, v_cache_param, v_new):
        '''
        This approximates a full attention module workload w/ loopback

        q : 1 x 32 x d
        k_cache_param: 1 x s x d
        k_new:  1 x 32 x d
        v_cache_param: 1 x s x d
        v_new: 1 x 32 x d
        '''
        _, s, d = k_cache_param.size()
        nh = self.nh
        dh = d // nh

        # Construct K matrix
        k = k_new.view(1, 32, nh, dh).permute(0, 2, 1, 3) # 1, nh, 32, dh
        k_cache = k_cache_param.view(1, s, nh, dh).permute(0, 2, 1, 3) # 1, nh, s, dh
        k = torch.cat((k_cache, k), dim=-2)

        # Swizzle Q
        q = q.view(1, 32, nh, dh).permute(0, 2, 1, 3) # 1, nh, 32, dh

        # Obtain attention probabilities
        attn = torch.matmul(q, k.transpose(-1, -2)) # 1, nh, 32, s+32
        probs = torch.nn.functional.softmax(attn, dim=-1)

        # Construct V matrix
        v = v_new.view(1, 32, nh, dh).permute(0, 2, 1, 3) # 1, nh, 32, dh
        v_cache = v_cache_param.view(1, s, nh, dh).permute(0, 2, 1, 3) # 1, nh, s, dh
        v = torch.cat((v_cache, v), dim=-2) # 1, nh, s+32, dh

        # Obtain output of attention
        out = torch.matmul(probs, v) # 1, nh, 32, dh

        k_ret = k[:,:,-32:,:].permute(0,2,1,3).reshape(1,32,d)
        v_ret = v[:,:,-32:,:].permute(0,2,1,3).reshape(1,32,d)

        return out, k_ret, v_ret

def test_attn_module_cache_loopback(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    s = 480
    d = 768
    nh = 12
    dh = d // nh 
    mod = AttentionModuleLoopback(nh)
    module = pybuda.PyTorchModule('attn_module_loopback_fractured', mod)

    df = pybuda.DataFormat.Float16
    pybuda.set_configuration_options(default_df_override=df,
        accumulate_df=df)

    from pybuda.config import _get_global_compiler_config
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.loopback_outputs = {"k_cache_param": 1, "v_cache_param": 2}


    # '''
    # Fracturing
    # '''
    factor = 2
    pybuda.config.insert_fracture_group([
        ("k_cache_param", -1, factor),
        ("concatenate_4", -3, factor),
        ("matmul_7", -3, factor),
        ("softmax_9", -3, factor),
        ("matmul_17", -3, factor),
    ])

    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=module)
    
    qkv_act_size = (1, 32, d)
    cache_size = (1, s, d)

    compile_inputs = (torch.randn(qkv_act_size), torch.randn(cache_size), torch.randn(qkv_act_size), torch.randn(cache_size), torch.randn(qkv_act_size))
    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_inputs,)

    inputs = (torch.randn(qkv_act_size), torch.randn(qkv_act_size), torch.randn(qkv_act_size))
    tt0.push_to_inputs(inputs)
    pybuda.run_generate(input_count=1, write_index=0)
