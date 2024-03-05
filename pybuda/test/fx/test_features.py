# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import pybuda
import torch
import torch.nn as nn
import os
from pybuda.torch_compile import compile_torch

def test_link():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32, bias=True)

        def forward(self, x1):
            m1 = self.linear(x1)
            return m1

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


# Clip-like argmax code that does argmax followed by index
class ClipArgmax(torch.nn.Module):
    def forward(self, last_hidden_state, input_ids):
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        return pooled_output

def test_clip_argmax():
    model = torch.compile(ClipArgmax(), backend=compile_torch)
    input = (torch.rand(1, 128, 768), torch.randint(0, 30000, (1, 128)))
    tt_input = (input[0].to('tt'), input[1].to('tt'))
    tt_res = model(*tt_input)
    tt_res = tt_res.to('cpu')

    cpu_res = ClipArgmax()(*input)
    assert cpu_res == tt_res

