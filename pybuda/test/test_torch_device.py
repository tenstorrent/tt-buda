# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import pybuda
import torch
import torch.nn as nn
import os
from transformers import BertModel, GPT2LMHeadModel, GPT2Config, GPT2Model
from pybuda.torch_compile import compile_torch
from typing import Tuple

def test_gpt2():
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 2

    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    compile_cfg.enable_link_past_cache_ios = True
    compile_cfg.cpu_fallback_ops = set()
    compile_cfg.default_df_override = pybuda._C.Float16_b

    gpt2 = GPT2LMHeadModel(config).eval()
    input_ids = torch.randint(0, 10000, (1, 32)).int()
    golden = gpt2(input_ids)

    pybuda_mod = torch.compile(gpt2, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)

    next_token_logits = result[0]
    next_token_logits = next_token_logits.to("cpu")

    res = result[0].to("cpu")
    assert pybuda.op.eval.compare_tensor_to_golden(f"gpt2", golden[0], res, is_buda=True, pcc=0.99)
    
def test_gen():
    pytest.skip()   # Working on it
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 1

    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    # compile_cfg.enable_link_past_cache_ios = True
    compile_cfg.cpu_fallback_ops = set()
    compile_cfg.default_df_override = pybuda._C.Float16_b

    gpt2 = GPT2LMHeadModel(config).eval()
    gpt2.to("tt")

    input_ids = torch.randint(0, 10000, (1, 32)).int().to("tt")
    # past_cache_shape = (1, 12, 96, 64)
    # past_cache = []
    # for _ in range(config.num_hidden_layers):
    #     past_cache.append((torch.zeros(past_cache_shape).to("tt"), torch.zeros(past_cache_shape).to("tt")))
    # past_cache = tuple(past_cache)

    pybuda_mod = torch.compile(gpt2, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)

    res = result[0].to("cpu")
    breakpoint()
    inp2 = torch.randint(0, 10000, (1, 32)).int()
    inp2 = inp2.to("tt")
    result = pybuda_mod(inp2, result[1])
    
def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return x1 + x2, x2 + x1 + 2

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = Add()
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    golden = model(*inputs)
    pybuda_mod = torch.compile(model, backend=compile_torch)
    # inputs = [i.to("tt") for i in inputs]
    result = pybuda_mod(*inputs)
    result = [r.to("cpu") for r in result]

    assert [torch.allclose(g, r) for g, r in zip(golden, result)]

def test_linear():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 64, bias=True)

        def forward(self, x1, x2):
            m1 = self.linear(x1)
            return m1 + x2

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = Linear()
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 64)]
    golden = model(*inputs)
    # inputs = [i.to("tt") for i in inputs]
    pybuda_mod = torch.compile(model, backend=compile_torch)
    result = pybuda_mod(*inputs)
    result = result.to("cpu")

    assert pybuda.op.eval.compare_tensor_to_golden(f"linear", golden, result, is_buda=True, pcc=0.99)

def test_bert():
    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    compile_cfg.cpu_fallback_ops = set()

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    bert_cpu = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)


    input_ids = torch.randint(0, 10000, (1, 128)).int()
    golden = bert_cpu(input_ids)

    print("Copying model")
    bert.to("tt")

    print("Copying inputs")
    input_ids = input_ids.to("tt")

    print("Compiling Model")
    pybuda_mod = torch.compile(bert, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)
    print("Copying outputs")

    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp2 = torch.randint(0, 10000, (1, 128)).int()
    golden = bert_cpu(inp2)

    inp2 = inp2.to("tt")
    result = pybuda_mod(inp2)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp3 = torch.randint(0, 10000, (1, 64)).int()
    golden = bert_cpu(inp3)
    inp3 = inp3.to("tt")
    result = pybuda_mod(inp3)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp4 = torch.randint(0, 10000, (1, 128)).int()
    golden = bert_cpu(inp4)
    inp4 = inp4.to("tt")
    result = pybuda_mod(inp4)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp5 = torch.randint(0, 10000, (1, 64)).int()
    golden = bert_cpu(inp5)
    inp5 = inp5.to("tt")
    result = pybuda_mod(inp5)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)


from torch._dynamo import export
from torch._decomp import register_decomposition
import torch
import torch.nn as nn

torch._dynamo.reset()
import torch._dynamo as dynamo


def test_decomp():
    pytest.skip() #TODO fix: FATAL    | Always          - Unsupported (for now) _copy_from TTDevice[0] to TTDevice[0]
    os.environ["PYBUDA_DEVMODE"] = "1"
    class BasicModule(nn.Module):
        def forward(self, x):
            x = x * 2
            a,b,c = torch.split(x, 3, dim=-1)
            return a + b + c

    mod, input = BasicModule(), torch.randn(2, 9).to(dtype=torch.float16)
 
    pybuda_mod = torch.compile(mod, backend=compile_torch, dynamic=False)
    out = pybuda_mod(input)

@pytest.mark.parametrize("shape", [(1024, 1024)])
@pytest.mark.parametrize("mb", [1, 8, 16])
@pytest.mark.parametrize("loop", [1, 8, 16])
@pytest.mark.parametrize("native", [True, False])
def test_push(shape, mb, loop, native):
    if mb != 1:
        pytest.skip() #TODO
    os.environ["PYBUDA_DEVMODE"] = "1"
    import time

    pybuda.config.set_configuration_options(
        default_df_override=pybuda.config.DataFormat.Float32
    )

    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return x1 + x2

    model = Add()
    sample_inputs = [torch.empty(mb, 1, *shape), torch.empty(mb, 1, *shape)]
    inputs = [(torch.ones(mb, 1, *shape), torch.ones(mb, 1, *shape))] * loop

    if native:
        model = model.to("tt")
        pybuda_mod = pybuda_mod = torch.compile(model, backend=compile_torch, dynamic=False)
        comp_inputs = [i.to("tt") for i in inputs[0]]
        result = pybuda_mod(*comp_inputs) # compile
        start = time.perf_counter()
        for args in inputs:
            args = [a.to("tt") for a in args]
            result = pybuda_mod(*args)
            result.to("cpu")
        elapsed = time.perf_counter() - start
    else:
        tt0 = pybuda.TTDevice("tt0")
        tt0.place_module(pybuda.module.PyTorchModule("add", model))
        output_q = pybuda.initialize_pipeline(
            training=False, sample_inputs=sample_inputs
        )

        start = time.perf_counter()
        for i in range(loop):
            tt0.push_to_inputs(inputs[i])
        pybuda.run_forward(input_count=loop)
        for i in range(loop):
            result = output_q.get(timeout=30)
        elapsed = time.perf_counter() - start

    float32_size = 4
    data = mb * shape[0] * shape[1] * float32_size / (1024 * 1024)

    print(
        f"Batch[{mb:2}] Loop[{loop:2}] Native[{native:1}] Data[{data}mB] Elapsed[{elapsed:2.4}sec]"
    )
