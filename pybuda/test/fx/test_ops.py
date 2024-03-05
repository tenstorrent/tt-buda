# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import pybuda
import torch
import torch.nn as nn
import os
from pybuda.torch_compile import compile_torch

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

def test_conv2d():
    class Conv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = Conv2d()
    inputs = torch.rand(1, 3, 32, 32)
    golden = model(inputs) 

    if True:
        pybuda_mod = torch.compile(model, backend=compile_torch, dynamic=False)
        result = pybuda_mod(inputs)
        result = result.to("cpu")
        assert pybuda.op.eval.compare_tensor_to_golden(f"conv2d", golden, result, is_buda=True, pcc=0.99)
    else: 
        from pybuda.verify.backend import verify_module
        mod = pybuda.PyTorchModule("conv", model)
        verify_module(
            mod,
            ([1,3,32,32],),
            verify_cfg=pybuda.VerifyConfig(
                arch=pybuda.BackendDevice.Wormhole_B0,
                devtype=pybuda.BackendType.Golden,
                test_kind=pybuda.verify.TestKind.INFERENCE,
                pcc=0.99
            ), 
        )

def test_bn():
    class BN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x):
            x = self.bn(x)
            return x

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = BN()
    model.eval()

    inputs = torch.rand(1, 64, 32, 32)
    golden = model(inputs)
    # inputs = [i.to("tt") for i in inputs]
    pybuda_mod = torch.compile(model, backend=compile_torch)
    result = pybuda_mod(inputs)
    result = result.to("cpu")

    assert pybuda.op.eval.compare_tensor_to_golden(f"linear", golden, result, is_buda=True, pcc=0.99)

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
    pybuda_mod = torch.compile(model.to("tt"), backend=compile_torch)
    result = pybuda_mod(*[i.to("tt") for i in inputs])
    result = result.to("cpu")

    assert pybuda.op.eval.compare_tensor_to_golden(f"linear", golden, result, is_buda=True, pcc=0.99)
