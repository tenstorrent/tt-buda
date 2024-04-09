# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
import os

import pytest
import torch
import torch.nn as nn

import pybuda
from pybuda.torch_compile import compile_torch
from pybuda.config import _get_global_compiler_config

from .conftest import generic_model_test

#
# TODO: Tests here depend on the fact that argmax/index are not supported at the moment. If that changes, and they are added to the device, many of
# these tests will be irrelevant, and need to be updated with a different fallback op. Ideally something that we would (almost) never support.
#

def test_link():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32, bias=True)

        def forward(self, x1):
            m1 = self.linear(x1)
            return m1

    _get_global_compiler_config().enable_pt2_fx_graph_link = True
    os.environ["PYBUDA_DEVMODE"] = "1"
    input = torch.rand(1, 32, 32)
    input2 = torch.rand(1, 32, 32)
    input3 = torch.rand(1, 32, 32)

    input = input.to("tt")

    pybuda_mod = torch.compile(Linear().to("tt"), backend=compile_torch)
    result_c = pybuda_mod(input)
    pybuda_mod_2 = torch.compile(Linear().to("tt"), backend=compile_torch)
    result__ = pybuda_mod_2(result_c)

    result_c = pybuda_mod(input)
    result = pybuda_mod_2(result_c)

    result = result.to("cpu")

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


# Clip-like argmax code that does argmax followed by index
class ClipArgmax(torch.nn.Module):
    def __init__(self, eltwise_before, eltwise_after):
        super().__init__()
        self.eltwise_before = eltwise_before
        self.eltwise_after = eltwise_after

    def forward(self, last_hidden_state, input_ids):
        if self.eltwise_before:
            last_hidden_state = last_hidden_state * last_hidden_state # something to do on device
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(device=last_hidden_state.device).argmax(dim=-1),
            ]
        if self.eltwise_after:
            pooled_output = pooled_output * pooled_output # something to do on device
        return pooled_output

@pytest.mark.parametrize("eltwise_before", [True, False])
@pytest.mark.parametrize("eltwise_after", [True, False])
def test_fallback(eltwise_before, eltwise_after):
    shape = (1, 128, 768)
    generic_model_test(ClipArgmax(eltwise_before, eltwise_after), inputs=(torch.rand(*shape), torch.randint(0, shape[1], (1, shape[1])).int()))

class ClipArgmaxSandwich(torch.nn.Module):
    def forward(self, last_hidden_state, input_ids):
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(device=last_hidden_state.device).argmax(dim=-1),
            ]
        pooled_output = pooled_output * pooled_output # something to do on device
        pooled_output = pooled_output[
                torch.arange(pooled_output.shape[0], device=last_hidden_state.device),
                input_ids.to(device=last_hidden_state.device).argmax(dim=-1),
            ]
        return pooled_output

def test_fallback_before_and_after():
    # Fallback before and after, with device in the middle
    shape = (1, 128, 768)
    generic_model_test(ClipArgmaxSandwich(), inputs=(torch.rand(*shape), torch.randint(0, shape[1], (shape[0], shape[1])).int()))


class RawIntOutput(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 128
        vocab_size = 1024
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        seq_length = input_ids.shape[-1]
        input_ids = input_ids[:, :seq_length]
        emb = self.token_embedding(input_ids)
        return emb, input_ids

def test_fallback_on_raw_int():
    # Test the case where the raw int output into embedding is also passed through to output, through some kind of nop/reshape/slice
    # We want to fall back to CPU for the raw int output
    generic_model_test(RawIntOutput(), inputs=[torch.randint(0, 1024, (1, 128)).int()])

class FallbackOutputReuse(nn.Module):
    def forward(self, a):
        b = a * a
        c = torch.argmax(b, dim=-1)
        return c, b

def test_fallback_with_output_reuse():
    # Test the case where the fallback graph is using one of the graph outputs as its input
    generic_model_test(FallbackOutputReuse(), num_outputs=2)

class ForkedInput(torch.nn.Module):
    def forward(self, last_hidden_state, input_ids):
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(device=last_hidden_state.device).argmax(dim=-1),
            ]
        device_output = last_hidden_state * last_hidden_state # something to do on device
        return pooled_output, device_output

def test_forked_input():
    # Test the case where the input is used in both fallback and device graph
    generic_model_test(ForkedInput(), inputs=(torch.rand(1, 128, 768), torch.randint(0, 128, (1, 128)).int()), num_outputs=2)

class ForkedInputToNop(torch.nn.Module):
    def forward(self, last_hidden_state, input_ids):
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(device=last_hidden_state.device).argmax(dim=-1),
            ]
        device_output = last_hidden_state 
        return pooled_output, device_output

def test_forked_input_to_nop():
    # Test the case where the input is used in both fallback and device graph, but device graph is NOP so it also falls back to CPU
    generic_model_test(ForkedInputToNop(), inputs=(torch.rand(1, 128, 768), torch.randint(0, 128, (1, 128)).int()), num_outputs=2)

foobar = 5.0
class DisjointedGraphs(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        a = a.to('cpu')
        if a[:, 0] > foobar:
            b = a + 2
        else:
            b = a + 3

        return b

def test_disjointed_graphs():
    # Test the case where pt2 generates two completely independent graphs
    generic_model_test(DisjointedGraphs(), inputs=(torch.Tensor([[4.0]]),))

class DisjointedGraphsWithParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1, bias=False)
        self.linear2 = torch.nn.Linear(1, 1, bias=False)
    def forward(self, a):
        a = self.linear1(a)
        a = a.to('cpu')
        if a[0] > 1:
            b = a + 2
        else:
            b = self.linear2(a) 

        return b

@pytest.mark.skip(reason="Fails in shape handling, Allan is working on it.. or we need to cause disjointed graphs differently")
def test_disjointed_graphs_with_params():
    generic_model_test(DisjointedGraphsWithParams(), inputs=(torch.tensor([4.0]),))

class ModelWithTensorAttributes(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self, x):
        return x + self.a

@pytest.mark.skip(reason="Input 0 for op add_0 is uninitialized, missing queue settings could cause us to access out of bounds queue.")
def test_model_with_attributes():
    # Test the case where the model has attributes that are used in the calculation
    shape = (32, 64)
    generic_model_test(ModelWithTensorAttributes(torch.rand(*shape).to('tt')), inputs=(torch.rand(*shape),))

class ModelWithTensorAttributesNoInput(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self):
        return self.a * 2

def test_model_with_attributes_no_input():
    # Test the case where the model has attributes that are used in the calculation
    shape = (32, 64)
    generic_model_test(ModelWithTensorAttributesNoInput(torch.rand(*shape)), num_inputs=0)

