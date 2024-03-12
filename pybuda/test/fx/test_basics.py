# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from tensorflow.security.fuzzing import py
import torch
from pybuda.torch_compile import compile_torch

class NoOutputGraph(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        a = 3 * a

def test_no_output_graph():
    model = torch.compile(NoOutputGraph(), backend=compile_torch)
    input = torch.tensor([[1.0]])
    tt_res = model(input.to('tt'))

    cpu_res = NoOutputGraph()(input)
    assert cpu_res == tt_res

class DanglingOps(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        b = a + 2
        c = b * 12
        return a

def test_dangling_ops():
    model = torch.compile(DanglingOps(), backend=compile_torch)
    input = torch.tensor([[1.0]])
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')

    cpu_res = DanglingOps()(input)
    assert cpu_res == tt_res

foobar = 5.0
class DisjointedGraphs(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        a = a.to('cpu')
        if a[0] > foobar:
            b = a + 2
        else:
            b = a + 3

        return b

def test_disjointed_graphs():
    model = torch.compile(DisjointedGraphs(), backend=compile_torch)
    input = torch.tensor([[1.0]])
    tt_res_ = model(input.to('tt'))
    tt_res_ = tt_res_.to('cpu')
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')
    cpu_res = DisjointedGraphs()(input)
    assert cpu_res == tt_res

    input = torch.tensor([[2.5]])
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')

    cpu_res = DisjointedGraphs()(input)
    assert cpu_res == tt_res

@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2428")
def test_to_double():
    tensor = torch.rand(32, 32).to('tt')
    tensor.to(dtype=torch.double)
 
@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2429")
def test_print():
    tensor = torch.rand(32, 32).to('tt')
    print(tensor)

@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2438")
def test_longint():
    original_data = torch.randint(0, 10, (1, 8))
    tensor = original_data.to('tt').to(dtype=torch.int).to('cpu')

    original_data = original_data.to(dtype=torch.int)
    assert torch.allclose(original_data, tensor)

