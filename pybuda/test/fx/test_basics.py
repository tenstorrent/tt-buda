# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from tensorflow.security.fuzzing import py
import torch
from pybuda.torch_compile import compile_torch

from .conftest import generic_model_test

class NoOutputModel(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        a = 3 * a

def test_no_output_graph():
    # Test the case where the model has no outputs
    generic_model_test(NoOutputModel(), num_outputs=0)

class NoInputModel(torch.nn.Module):
    def forward(self):
        return torch.tensor([1])

@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2475")
def test_no_input_model():
    # Test the case where the model has no inputs
    generic_model_test(NoInputModel(), num_inputs=0)

class EmptyModelNoOutput(torch.nn.Module):
    def forward(self, a):
        pass

def test_empty_model_no_output():
    # Test the case where the model has no operations, and no output
    generic_model_test(EmptyModel())

class EmptyModel(torch.nn.Module):
    def forward(self, a):
        return a

def test_empty_model():
    # Test the case where the model has no operations
    generic_model_test(EmptyModel())

class DanglingOps(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        b = a + 2
        c = b * 12
        return a

def test_dangling_ops():
    # Test the case where the model has an op who output goes nowhere
    generic_model_test(DanglingOps())


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


class NonAlignedSize(torch.nn.Module):
    def forward(self, a):
        return a + 1

@pytest.mark.skip(reason="These tests fail when running all of test_basics, but pass by themselves. There's some kind of corruption going on.")
@pytest.mark.parametrize("rows", [1, 32])
def test_return_non_aligned_sizes(rows):
    model = torch.compile(NonAlignedSize(), backend=compile_torch)
    input = torch.rand(1, rows, 33)
    input_tt = input.to('tt')
    tt_res = model(input_tt).to('cpu')
    cpu_res = NonAlignedSize()(input)
    assert torch.allclose(cpu_res, tt_res, atol=0, rtol=1e-3)

