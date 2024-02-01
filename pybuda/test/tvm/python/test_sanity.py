# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest
import sys

import torch
import tensorflow as tf
import torch.nn as nn
import pybuda
from pybuda.tvm_to_python import compile_tvm_to_python
from pybuda import (
    TTDevice,
    BackendDevice,
    BackendType,
    pybuda_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
    Tensor,
    TFModule,
    CompileDepth,
)
from transformers import BertModel, BertConfig
from transformers import GPT2Model, GPT2Config
from collections import OrderedDict
from pybuda.op.eval import compare_tensor_to_golden

import importlib
from test.tvm.utils import evaluate_framework_vs_pybuda
from test.utils import download_model


def test_linear():
    
    class DoubleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(64, 64, bias=True)
            self.l2 = nn.Linear(64, 64, bias=True)

        def forward(self, x1):
            m1 = self.l1(x1)
            m1g = nn.functional.gelu(m1)

            m2 = self.l2(m1g)
            m2g = nn.functional.gelu(m2)
            return m2g

    model = DoubleLinear()
    mod = PyTorchModule("double_linear", model)

    shape = (128, 64)
    act1 = torch.rand(shape)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        tvm_module_to_num_patterns={mod.get_name(): 2}
        # tvm_graph_load_path="linear.json",
        # tvm_graph_store_path="linear.json", 
        )

    test_name = "double_linear"
    module_writers, _ = compile_tvm_to_python(mod, test_name, (act1,), compiler_cfg=compiler_cfg)
    writer = module_writers[0]

    sys.path.append(".")
    TestClass = getattr(importlib.import_module((writer.import_module_path())), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(model)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    act1_buda = Tensor.create_from_torch(act1)

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, act1_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(model, res, act1)


def test_linear_tf():
    
    class DoubleLinear(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(64, use_bias=True)
            self.dense2 = tf.keras.layers.Dense(64, use_bias=True)

        def call(self, x1):
            m1 = self.dense1(x1)
            m2 = self.dense2(m1)

            return m1 + m2

    model = DoubleLinear()
    mod = TFModule("double_linear_tf", model)

    shape = (128, 64)
    act1 = tf.random.uniform(shape)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        # tvm_graph_load_path="linear.json",
        # tvm_graph_store_path="linear.json", 
        )

    test_name = "double_linear_tf"
    module_writers, _ = compile_tvm_to_python(mod, test_name, (act1,), compiler_cfg=compiler_cfg)
    writer = module_writers[0]
    
    sys.path.append(".")
    TestClass = getattr(importlib.import_module(writer.import_module_path()), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(model)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    act1_buda = Tensor.create_from_torch(torch.tensor(act1.numpy()))

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, act1_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(model, res, act1)


def test_bert():
    num_layers = 12
    # model = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny", torchscript=True)
    config.num_hidden_layers = num_layers
    model = BertModel(config)

    shape = (1, 128, 128)
    hidden_states = torch.rand(*shape)

    submodel = model.encoder
    mod = PyTorchModule("bert_encoder", submodel)
    pytorch_out = submodel(hidden_states)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        tvm_module_to_num_patterns={mod.get_name(): num_layers}
        # tvm_graph_load_path="bert.json",
        # tvm_graph_store_path="bert.json",
        # match_subgraph_patterns=num_layers,
        )

    test_name = "bert"
    writers, _ = compile_tvm_to_python(mod, test_name, (hidden_states,), compiler_cfg=compiler_cfg)
    writer = writers[0]

    sys.path.append(".")
    TestClass = getattr(importlib.import_module(writer.import_module_path()), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(submodel)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    hidden_states_buda = Tensor.create_from_torch(hidden_states)

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, hidden_states_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(submodel, res, hidden_states)


def test_gpt2():
    num_layers = 2
    config = GPT2Config()
    model = GPT2Model(config)

    shape = (1, 64, 768)
    hidden_states = torch.rand(*shape)

    submodel = model.h[0]
    mod = PyTorchModule("gpt2", submodel)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        tvm_module_to_num_patterns={mod.get_name(): num_layers}
        # tvm_graph_load_path="gpt2.json",
        # tvm_graph_store_path="gpt2.json",
        # match_subgraph_patterns=num_layers,
        # compile_depth=CompileDepth.POST_PATTERN_MATCHER,
        )

    test_name = "gpt2"
    writers, _ = compile_tvm_to_python(mod, test_name, (hidden_states,), compiler_cfg=compiler_cfg)
    writer = writers[0]

    sys.path.append(".")
    TestClass = getattr(importlib.import_module(writer.import_module_path()), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(submodel)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    hidden_states_buda = Tensor.create_from_torch(hidden_states)

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, hidden_states_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(submodel, res, hidden_states)


def test_gpt2_multiple_layers():
    class GPT2MultipleLayers(nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            config = GPT2Config()

            self.layer_list = GPT2Model(config).h[:num_layers]
            self.num_layers = num_layers

        def forward(self, hidden_states):
            for i in range(self.num_layers):
                hidden_states = self.layer_list[i](hidden_states)[0]
            return hidden_states

    num_layers = 2
    model = GPT2MultipleLayers(num_layers)

    shape = (1, 64, 768)
    hidden_states = torch.rand(*shape)

    mod = PyTorchModule("gpt2_multi", model)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        tvm_module_to_num_patterns={mod.get_name(): num_layers}
        # tvm_graph_load_path="gpt2.json",
        # tvm_graph_store_path="gpt2.json",
        # match_subgraph_patterns=num_layers,
        # compile_depth=CompileDepth.POST_PATTERN_MATCHER,
        )

    test_name = "gpt2_multi"
    writers, _ = compile_tvm_to_python(mod, test_name, (hidden_states,), compiler_cfg=compiler_cfg)
    writer = writers[0]

    sys.path.append(".")
    TestClass = getattr(importlib.import_module(writer.import_module_path()), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(model)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    hidden_states_buda = Tensor.create_from_torch(hidden_states)

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, hidden_states_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(model, res, hidden_states)

def test_resnet():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    mod = PyTorchModule("resnet", model)

    shape = (1, 3, 320, 320)
    act1 = torch.rand(shape)
    compiler_cfg = CompilerConfig(enable_training=False,)
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    test_name = "resnet"

    writers, _ = compile_tvm_to_python(mod, test_name, (act1,), compiler_cfg=compiler_cfg)
    writer = writers[0]

    sys.path.append(".")
    TestClass = getattr(importlib.import_module(writer.import_module_path()), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(model)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    act1_buda = Tensor.create_from_torch(act1)

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, act1_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(model, res, act1)


def test_unsupported():
    class DoubleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(64, 64, bias=False)
            self.l2 = nn.Linear(64, 64, bias=False)

        def forward(self, x1):
            m1 = self.l1(x1)
            m1g = nn.functional.gelu(m1)

            m2 = self.l2(m1g)
            m2g = torch.acos(m2)
            return m2g

    n_loops = 2

    model = DoubleLinear()
    mod = PyTorchModule("unsupported", model)

    shape = (128, 64)
    act1 = torch.rand(shape)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        enable_tvm_unsupported_ops=True,
        )

    test_name = "unsupported"
    writers, _ = compile_tvm_to_python(mod, test_name, (act1,), compiler_cfg=compiler_cfg)

def test_bert_base():
    num_layers = 2
    config = download_model(BertConfig.from_pretrained, "bert-base-uncased", torchscript=True)
    config.num_hidden_layers = num_layers
    model = BertModel(config)

    shape = (1, 128, 768)
    hidden_states = torch.rand(*shape)
    attention_mask = torch.rand(1, 1, 1, 128) # extended_attention_mask

    submodel = model.encoder
    test_name = "bert_encoder_attn_mask"
    mod = PyTorchModule(test_name, submodel)
    pytorch_out = submodel(hidden_states, attention_mask)

    compiler_cfg = CompilerConfig(
        enable_training=False,
        tvm_module_to_num_patterns={mod.get_name(): num_layers}
    )

    writers, _ = compile_tvm_to_python(mod, test_name, (hidden_states, attention_mask), compiler_cfg=compiler_cfg)
    writer = writers[0]

    sys.path.append(".")
    TestClass = getattr(importlib.import_module(writer.import_module_path()), writer.class_name)

    buda_mod = TestClass("test_module")

    buda_mod.process_framework_parameters(submodel)

    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=buda_mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(buda_mod)

    hidden_states_buda = Tensor.create_from_torch(hidden_states)
    attention_mask_buda = Tensor.create_from_torch(attention_mask)

    sgd_optimizer.set_optimizer_parameters()

    res = pybuda_compile(tt0, test_name, hidden_states_buda, attention_mask_buda, compiler_cfg=compiler_cfg, verify_cfg=VerifyConfig())

    evaluate_framework_vs_pybuda(submodel, res, hidden_states)


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
                
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip("Wait until #1005 is resolved")

    s = 480
    d = 24*1024
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
        ("v_cache_param", -1, factor),
        ("concatenate_13", -3, factor),
        ("concatenate_4", -3, factor),
        ("matmul_7", -3, factor),
        ("softmax_9", -3, factor),
        ("matmul_17", -3, factor),
    ])

    # pybuda.config.insert_fracture_group([

    #     # ("matmul_7", -3, factor),
    #     # ("softmax_9", -3, factor),
    #     # ("matmul_17", -3, factor),
    # ])

    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=module)
    
    qkv_act_size = (1, 32, d)
    cache_size = (1, s, d)

    compile_inputs = (torch.randn(qkv_act_size), torch.randn(cache_size), torch.randn(qkv_act_size), torch.randn(cache_size), torch.randn(qkv_act_size))
    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_inputs,)

    inputs = (torch.randn(qkv_act_size), torch.randn(qkv_act_size), torch.randn(qkv_act_size))
    tt0.push_to_inputs(inputs)
    pybuda.run_generate(input_count=1, write_index=0)


def test_scheduler_write_back_parameters(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class MinimalRepro(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(64, 64)
            self.fc2 = torch.nn.Linear(64, 64)

        def forward(self, x, y_new, y_cache):
            # x, y_cache : (1, 1, 64, 64)
            y_cache = torch.cat((y_cache, y_new), dim=-2)
            x = x + y_cache
            x = self.fc1(x)
            x = self.fc2(x)

            y_slice = y_cache[:,:,-32:, :]

            return x, y_slice


    model = MinimalRepro()
    x = torch.rand(1, 1, 64, 64)
    y_cache = torch.rand(1, 1, 32, 64)
    y_new = torch.randn(1, 1, 32, 64)

    compile_inputs=(x, y_new, y_cache)
    inputs=(x, y_cache)
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.loopback_outputs = {'y_cache_1': 1}
    
    tt = pybuda.TTDevice('tt0', devtype=test_device.devtype, arch=test_device.arch)
    tt.place_module(pybuda.PyTorchModule('min_repro', model))

    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_inputs)
    tt.push_to_inputs(inputs)
    pybuda.run_generate(input_count=1, write_index=0)

    out = output_q.get()