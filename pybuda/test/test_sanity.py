# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
import yaml
import random
import os
import queue

from collections import defaultdict 
from typing import Dict, List
import pybuda
import pybuda.op
from pybuda import (
    PyBudaModule,
    TTDevice,
    BackendDevice,
    BackendType,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
    PyTorchModule,
)
from pybuda.ttdevice import get_device_config
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.utils import align_up_tile
from pybuda.pybudaglobal import TILE_DIM
from pybuda.op.eval import compare_tensor_to_golden
from .common import compile, device, run, run_torch, ModuleBuilder
import pybuda.verify as verify
from pybuda.verify import TestKind, verify_module
from test.bert.modules import PyBudaBertMHA, get_bert_parameters

verify_cfg = VerifyConfig(run_golden=True, run_net2pipe=True) # Run backend golden check on all tests in here

backend_devices = {
    "grayskull" : BackendDevice.Grayskull,
    "wormhole" : BackendDevice.Wormhole,
    "wormhole_b0": BackendDevice.Wormhole_B0,
}

class BudaTestAdd(PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (1, 1, 32, 32)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        a1 = pybuda.op.Add("add1", act1, self.weights1)
        a2 = pybuda.op.Add("add2", act2, self.weights2)
        a3 = pybuda.op.Add("add3", a1, a2)
        return a3

class BudaTest(PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        m1e = pybuda.op.Exp("exp", m1)
        return pybuda.op.Add("add", m1e, m2)


class MatmulModule(PyBudaModule):
    """
    Single Matmul module for basic testing
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name, bias):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.bias = bias
        if bias:
            bias_shape = (1, 1, 1, self.shape[-1])
            self.bias1 = pybuda.Parameter(*bias_shape, requires_grad=True)

    def forward(self, act1):
        if self.bias:
            m1 = pybuda.op.Matmul("matmul1", act1, self.weights1) + self.bias1
        else:
            m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        return m1


class SimpleLinear(PyBudaModule):
    """
    Linear module for basic testing
    """
    def __init__(self, name, in_features, out_features, bias=True, relu=True):
        super().__init__(name)
        self.weights = pybuda.Parameter(1, 1, in_features, out_features, requires_grad=True)
        sqrt_k = (1.0 / in_features) ** 0.5
        weights_value = torch.empty(*self.weights.shape.get_pytorch_shape(), requires_grad=True)
        torch.nn.init.uniform_(weights_value, -sqrt_k, sqrt_k)
        self.set_parameter("weights", weights_value)
        self.bias = None
        if bias:
            self.bias = pybuda.Parameter(out_features, requires_grad=True)
            bias_value = torch.empty(*self.bias.shape.get_pytorch_shape(), requires_grad=True)
            torch.nn.init.uniform_(bias_value, -sqrt_k, sqrt_k)
            self.set_parameter("bias", bias_value)
        self.relu = relu

    def forward(self, activations):
        x = pybuda.op.Matmul(self.name + ".matmul", activations, self.weights)
        if self.bias is not None:
            x = pybuda.op.Add(self.name + ".bias", x, self.bias)
        if self.relu:
            x = pybuda.op.Relu(self.name + ".relu", x)
        # TODO: fixme, problems if relu is the final graph output
        x = pybuda.op.Identity(self.name + ".ident", x)
        return x


def test_trace(training):

    mod = BudaTest("test_module")
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))
    act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))
    mod.set_parameter("weights2", torch.rand(*BudaTest.shape, requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    pybuda_compile(tt0, "sanity", act1, act2, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)


def test_trace_add_params(training):

    mod = BudaTest("test_module")
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))
    act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))
    mod.set_parameter("weights2", torch.rand(*BudaTest.shape, requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    pybuda_compile(tt0, "add_params", act1, act2, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)

@pytest.mark.parametrize("bias", (True, False), ids=["bias", "no_bias"])
def test_trace_matmul(training, bias):

    if bias:
        pytest.skip() # golden random fail in CI, to be figured out

    mod = MatmulModule("test_module", bias=bias)
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))
    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))

    if bias:
        shape = (1, 1, 1, BudaTest.shape[-1])
        mod.set_parameter("bias1", torch.rand(shape, requires_grad=True))

    sgd_optimizer.set_optimizer_parameters()

    pybuda_compile(tt0, "trace_matmul", act1, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)


def test_trace_log(test_kind):
    class BudaLogModule(PyBudaModule):
        shape = (1, 1, 64, 64)

        def __init__(self, name):
            super().__init__(name)

        def forward(self, act1):
            return pybuda.op.Log("log", act1)

    verify.verify_module(BudaLogModule("log_module"), [(1, 1, 64, 64)], VerifyConfig(
        graph_name="log", test_kind=test_kind, devtype=BackendType.NoBackend))

    """
    mod = BudaLogModule("log_module")
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(
        torch.rand(*BudaLogModule.shape, requires_grad=True)
    )
    sgd_optimizer.set_optimizer_parameters()
    pybuda_compile(tt0, "log", act1, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)
    """

def test_trace_add():
    class BudaAddModule(PyBudaModule):
        shape = (1, 1, 128, 128)

        def __init__(self, name):
            super().__init__(name)

        def forward(self, act1, act2):
            return pybuda.op.Add("add", act1, act2)

    verify.verify_module(BudaAddModule("add_module"), [(1, 1, 128, 128), (1, 1, 128, 128)], VerifyConfig())

def test_trace_constant():
    class BudaAddModule(PyBudaModule):
        shape = (1, 1, 128, 128)

        def __init__(self, name):
            super().__init__(name)

        def forward(self, act1):
            constant = pybuda.op.Constant("constant", constant=2.0)
            return pybuda.op.Add("add", act1, constant)

    mod = BudaAddModule("add_module")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(
        torch.rand(*BudaAddModule.shape, requires_grad=True)
    )

    vcfg = VerifyConfig(run_golden=False) # segfaults, need to skip running golden
    pybuda_compile(
        tt0,
        "constant",
        act1,
        compiler_cfg=CompilerConfig(enable_training=False),
        verify_cfg=vcfg,
    )


@pytest.mark.parametrize("mode", ["inference", "training", "optimizer"])
def test_trace_linear_relu(mode):
    training = (mode == "training" or mode == "optimizer")
    optimizer = (mode == "optimizer")
    mod = SimpleLinear("simple_linear", 64, 32, bias=False, relu=True)
    sgd_optimizer = pybuda.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    ) if optimizer else None
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)
    activations = Tensor.create_from_torch(torch.normal(0.0, 1.0, (1, 1, 128, 64), requires_grad=True))
    if sgd_optimizer is not None:
        sgd_optimizer.set_optimizer_parameters()
        
    verify_cfg=VerifyConfig(run_golden=False) # relu not supported yet
    pybuda_compile(
        tt0,
        "linear_relu",
        activations,
        compiler_cfg=CompilerConfig(enable_training=training),
        verify_cfg=verify_cfg
    )


@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("shapes", [
    ((1, 1, 32, 64), (1, 1, 64, 32)),
])
def test_reshape(mode, shapes):
    training = mode == "training"

    @compile(
        compiler_cfg=CompilerConfig(enable_training=training, compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER),
        verify_cfg=VerifyConfig(run_golden=False),  # reshape not supported by backend
    )
    def simple_reshape(x):
        x = pybuda.op.Identity("id0", x)
        x = pybuda.op.Reshape("reshape0", x, shapes[1])
        return pybuda.op.Identity("id1", x)

    x = Tensor.create_from_torch(torch.rand(*shapes[0], requires_grad=training))
    simple_reshape(x)

@pytest.mark.parametrize("type", ["Avg", "Sum"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("shapes", [
    ((1, 2, 32, 64), (1, 3, 64, 32)),
    ((1, 2, 64, 32), (1, 3, 32, 64)),
])
def test_reduce(training, shapes, type, dim, test_device):

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_reduce(x):
        if type == "Avg":
            x = pybuda.op.ReduceAvg("reduce", x, dim)
        else:
            x = pybuda.op.ReduceSum("reduce", x, dim)
        return x

    x = Tensor.create_from_torch(torch.rand(*shapes[0], requires_grad=training))
    simple_reduce(x)


@pytest.mark.parametrize("shape", [(1, 10, 32, 3072), (1, 10, 384, 1536), (1, 1, 512, 768)])
@pytest.mark.parametrize("dim_index_length", [(2, 0, 32, 0), (3, 64, 64, 0),(3, 64, -1, 0), (1, 9, 1, 0), (2, 0, 352, 0), (3, 0, 64, 192)])
def test_select(test_kind, test_device, shape, dim_index_length):
    dim, index, length, stride = dim_index_length
    if index + length > shape[dim]:
        pytest.skip()

    if length == -1:
        length = shape[dim] - index

    compiler_cfg = _get_global_compiler_config()
    #if test_kind.is_training():
    #    compiler_cfg.compile_depth = CompileDepth.BALANCER_PASS
    
    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, verify_all=True),
    )
    def simple_select(x):
        x0 = pybuda.op.Select("select0", x, dim, (index, length), stride=stride)
        x1 = pybuda.op.Select("select1", x, dim, (index, length), stride=stride)
        return pybuda.op.Multiply("mul0", x0, x1)

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=test_kind.is_training()))
    simple_select(x)

@pytest.mark.parametrize("shape", [(1, 3, 288, 124),(1, 6, 288, 124)])
@pytest.mark.parametrize("dim_index_length", [(-3, 1, 1, 2),])
def test_single_select(test_kind, test_device, shape, dim_index_length):
    dim, index, length, stride = dim_index_length
    if index + length > shape[dim]:
        pytest.skip()

    if length == -1:
        length = shape[dim] - index

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.manual_t_streaming = True
    # pybuda.config.override_t_stream_shape("index.dc.select.0", (9, 1))

    @compile(
        compiler_cfg = CompilerConfig(enable_t_streaming=True, manual_t_streaming = True),
        verify_cfg = VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, verify_all=True),
    )

    def simple_select(x):
        ret = pybuda.op.Select("select0", x, dim, (index, length), stride=stride)
        return ret

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=test_kind.is_training()))
    simple_select(x)

@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("shape", [(1, 1, 384, 384), (1, 12, 384, 384), (1, 1, 384, 96)])
@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("direction", ["h", "v"])
def test_slice_stack(mode, shape, factor, direction):
    training = mode == "training"

    slice_op = {"h": pybuda.op.HSlice, "v": pybuda.op.VSlice}[direction]
    stack_op = {"h": pybuda.op.HStack, "v": pybuda.op.VStack}[direction]

    verify_cfg=VerifyConfig(run_golden=False) # select not supported yet
    @compile(
        compiler_cfg=CompilerConfig(enable_training=training),
        verify_cfg=verify_cfg,
    )
    def simple_slice_stack(x):
        x = slice_op("slice", x, factor)
        x = stack_op("stack", x, factor)
        return x

    x = Tensor.create_from_torch(torch.rand(*shape, requires_grad=training))
    r = simple_slice_stack(x)
    assert torch.allclose(x.value(), r.outputs[0].value())


def test_trace_add_sub_rsub():
    class BudaAddSubRSubModule(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            self.one = pybuda.Parameter(1, requires_grad=True)
            self.set_parameter("one", torch.tensor((1.0,), requires_grad=False))

        def forward(self, act1, act2):
            two = self.one + self.one
            a = two + act1
            b = a + a
            c = act2 - b
            d = c - 3
            e = 4 - d
            f = 4 * e
            return f

    class TorchAddSubRSubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("one", torch.tensor((1.0,), requires_grad=False))

        def forward(self, act1, act2):
            two = self.one + self.one
            a = two + act1
            b = a + a
            c = act2 - b
            d = c - 3
            e = 4 - d
            f = 4 * e
            return f


    shape = (1, 1, 128, 128)
    mod = BudaAddSubRSubModule("add_sub_rsub")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)


    act1 = torch.rand(*shape)
    act2 = torch.rand(*shape)

    vcfg = VerifyConfig()
    ret = pybuda_compile(
        tt0,
        "add_sub_rsub",
        Tensor.create_from_torch(act1),
        Tensor.create_from_torch(act2),
        compiler_cfg=CompilerConfig(enable_training=False),
        verify_cfg=vcfg,
    )

    torchmod = TorchAddSubRSubModule()
    pytorch_out = torchmod(act1, act2)
    assert torch.allclose(pytorch_out, ret.outputs[0].value())


@pytest.mark.parametrize("dim", [None, 1, 2, -1])
def test_argmax(dim):
    verify_cfg=VerifyConfig(run_golden=False) # argmax not supported
    @compile(
        compiler_cfg=CompilerConfig(enable_training=False, compile_depth=CompileDepth.POST_INITIAL_GRAPH_PASS),
        verify_cfg=verify_cfg,
    )
    def simple_argmax(x):
        return pybuda.op.Argmax("argmax0", x, dim=dim)

    x = Tensor.create_from_torch(torch.rand((1, 2, 384, 384), requires_grad=False))
    simple_argmax(x)

@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("input_shape", [(1,1,1,32), (1,1,3,32)])
@pytest.mark.parametrize("max_value", [0.5, 0, 1, 5])
def test_argmax_multiple_maximums(dim, input_shape, max_value):
    verify_cfg=VerifyConfig(run_golden=False) # argmax not supported
    x = torch.zeros(input_shape)
    for i in range(input_shape[0]):
        x[0,0,i,2] = max_value
        x[0,0,i,4] = max_value
        x[0,0,i,6] = max_value
    x = Tensor.create_from_torch(x)
    @run(
        verify_cfg=verify_cfg,
    )
    def simple_argmax(x):
        return pybuda.op.Argmax("argmax0", x, dim=dim)

    simple_argmax(x)

def test_passthrough():

    @compile(compiler_cfg=CompilerConfig(enable_training=False))
    def passthrough(x1, x2):
        return x1 + 7, x2

    x1 = Tensor.create_from_torch(torch.rand((1, 64, 64)))
    x2 = Tensor.create_from_torch(torch.rand((1, 64, 64)))
    passthrough(x1, x2)

@pytest.mark.parametrize("dim", [1, 2, -1])
def test_max(test_kind, test_device, dim):
    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
        uniform_inputs=True,
        inputs_centered_on_zero=True,
        input_params=[{"data_format": torch.bfloat16}],
    )
    def simple_max(x):
        return pybuda.op.ReduceMax("max0", x, dim=dim)

    x = Tensor.create_from_torch(torch.randn((1, 4, 128, 128), requires_grad=test_kind.is_training()))
    simple_max(x)


@pytest.mark.parametrize("dim", [2, -1])
def test_reduce_tile_broadcast(test_kind, test_device, dim):
    pytest.skip("tenstorrent/pybuda#131")
    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
        uniform_inputs=True,
        inputs_centered_on_zero=True,
    )
    def simple_reduce_tile_broadcast(a, b):
        a = pybuda.op.ReduceMax("", a, dim=dim)
        return pybuda.op.Add("", a, b)

    a = Tensor.create_from_torch(torch.randn((1, 4, 4, 4), requires_grad=test_kind.is_training()))
    b = Tensor.create_from_torch(torch.randn((1, 4, 4, 4), requires_grad=test_kind.is_training()))
    simple_reduce_tile_broadcast(a, b)


@pytest.mark.parametrize("dim", [1, 2, -1])
@pytest.mark.parametrize("aligned", [True, False])
def test_concat(test_kind, test_device, dim, aligned):
    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def simple_concat(a, b):
        return pybuda.op.Concatenate("", a, b, axis=dim)
 
    if aligned:
        shapes = {
            -1: (1, 3, 128, 96),
            2: (1, 3, 1024, 32),
            1: (1, 1, 128, 32),
        }
        a = Tensor.create_from_torch(torch.randn((1, 3, 128, 32), requires_grad=test_kind.is_training()))
    else:
        shapes = {
            -1: (1, 3, 128, 6),
            2: (1, 3, 128, 6),
            1: (1, 1, 128, 6),
        }
        a = Tensor.create_from_torch(torch.randn((1, 3, 128, 6), requires_grad=test_kind.is_training()))
    b = Tensor.create_from_torch(torch.randn(shapes[dim], requires_grad=test_kind.is_training()))
    c = simple_concat(a, b)


class MultiEpochModule(pybuda.PyBudaModule):
    def __init__(self, name: str, num_matmuls: int):
        super().__init__(name)
        self.num_matmuls = num_matmuls
        self.weights = [pybuda.Parameter(64, 64, name = f"weights_{i}") for i in range(self.num_matmuls)]

    def forward(self, act):

        val = act
        for i in range(self.num_matmuls):
            val = pybuda.op.Matmul(f"matmul_{i}", val, self.weights[i])
            val = pybuda.op.Gelu(f"gelu_{i}", val)
            val = pybuda.op.Matmul(f"second_matmul_{i}", val, self.weights[i])

        return val

@pytest.mark.skip(reason="Backend doesn't support 'add' gradient op")
def test_recompute(test_device):

    microbatch_size = 1
    num_epochs = 3
    num_matmuls = 2 * num_epochs
    verify_module(MultiEpochModule("multi_epoch_module", num_matmuls), [(microbatch_size, 64, 64)],
            VerifyConfig(test_kind=TestKind.TRAINING_RECOMPUTE, devtype=test_device.devtype, arch=test_device.arch, 
                epoch_breaks=[f"matmul_{i}" for i in range(0, num_matmuls, 2)]))


@pytest.mark.parametrize("config", ["3x3conv", "data_mismatch", "c_stream", "in_out_stream"])
def test_sparse_matmul(test_device, config):
    from pybuda.op.eval.sparse_utils import create_conv2d_sparse_picker_matrix

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    if config == "3x3conv":
        iH, iW = (64, 64)
        inC = 32
        kH, kW = (3, 3)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "data_mismatch":
        minimal_tiles = 2
        act = torch.randn(32*minimal_tiles,32).unsqueeze(0).unsqueeze(0)
        out_tiles = minimal_tiles // 2
        eye = torch.eye(32*minimal_tiles, 32*minimal_tiles)
        pickers = [
            eye[:(out_tiles*32), :].to_sparse(),
            eye[(out_tiles*32-16):-16, :].to_sparse(),
        ]
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "c_stream":

        pytest.skip() # tenstorrent/budabackend#1543
        pybuda.config.override_t_stream_dir("sparse0.lc2", "C")
        pybuda.config.override_t_stream_shape("sparse0.lc2", (1, 32))
        iH, iW = (64, 64)
        inC = 1024
        kH, kW = (1, 1)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "in_out_stream":
        pybuda.config.override_t_stream_dir("buf0", "R")
        pybuda.config.override_t_stream_shape("buf0", (2, 1))
        pybuda.config.override_t_stream_dir("sparse0.lc2", "R")
        pybuda.config.override_t_stream_shape("sparse0.lc2", (3, 1))

        iH, iW = (32, 32)
        inC = 32
        kH, kW = (3, 3)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    else:
        raise RuntimeError("Unknown config")

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_sparse_matmul(act, sparse=None):
        if config == "in_out_stream":
            act = pybuda.op.Buffer("buf0", act)
        return pybuda.op.SparseMatmul("sparse0", sparse, act)

    simple_sparse_matmul(act, sparse=sparse)


def test_simple_clip(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_clip(act):
        return pybuda.op.Clip("clip0", act, 0.3, 0.7)

    act = Tensor.create_from_torch(torch.rand(1, 1, 32, 32))
    simple_clip(act)


@pytest.mark.parametrize("scheduler_policy", ["ModuleInputsBFS", "LongestPath"])
def test_deterministic_netlist(scheduler_policy):
    hidden_dim, num_heads, seq_len = (128, 4, 128)
    microbatch_size = 1

    reference_netlist = None

    for i in range(5):
        params = get_bert_parameters("mha", hidden_dim=hidden_dim)
        for _, param in params.items():
            t = torch.normal(mean=0.0, std=0.1, size=param.shape.get_pytorch_shape(), dtype=param.pt_data_format)
            param.set_value(t)

        config =  { "num_heads": num_heads, "encoder_index": 0 }
        mod = PyBudaBertMHA("mha", params, config)
        tt0 = TTDevice("tt0", devtype=BackendType.Golden)
        tt0.place_module(mod)

        verify_cfg = VerifyConfig(
                devtype=BackendType.Golden, 
                waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"}
        )
        act1 = Tensor.create_from_torch(torch.rand((microbatch_size, seq_len, hidden_dim)))
        act2 = Tensor.create_from_torch(torch.rand((microbatch_size, 1, seq_len)))
        ret = pybuda_compile(tt0, f"mha_{i}", act1, act2, compiler_cfg=CompilerConfig(enable_training=True, scheduler_policy=scheduler_policy), verify_cfg=verify_cfg)

        with open(ret.netlist_filename) as fd:
            netlist = yaml.safe_load(fd)
            if reference_netlist:
                assert reference_netlist == netlist, f"Expect netlist {i-1} to matche with netlist {i}"
            reference_netlist = netlist


class PadTest(PyBudaModule):
    """
    Test wrapper for pad
    """

    def __init__(self, name, pad, mode, channel_last):
        super().__init__(name)
        self.pad = pad
        self.channel_last = channel_last
        self.mode = mode

    def forward(self, act):
        return pybuda.op.Pad("pad", act, self.pad, self.mode, self.channel_last)


@pytest.mark.parametrize("shape", ([1, 1, 64, 64], [128, 768]), ids=["shape1x1x64x64", "shape128x768"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("pad", ([5, 11], [6, 13, 4, 17]), ids=["5x11", "6x13x4x17"])
@pytest.mark.parametrize("channel_last", (True, False), ids=["ChannelLast", "NotChannelLast"])
@pytest.mark.parametrize("mode", ["constant", "replicate"])
def test_pad(
    training,
    test_device, 
    recompute, 
    pad,
    shape,
    channel_last,
    mode,
):

    if len(shape) == 2 and channel_last:
        pytest.skip("input is supposed to be dim >= 3 to be channel-last")

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    if ((len(shape) == 2 and len(pad) == 4) or (len(shape) == 4 and len(pad) == 2)) and mode == "replicate":
        pytest.skip("PyTorch does not support the shape and pad combination")

    if mode == "replicate" and channel_last:
        pytest.skip()

    mod = PadTest(name="test_pad", pad=pad, mode=mode, channel_last=channel_last)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(shape, requires_grad=True))

    pybuda_compile(
        tt0,
        "pad",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER,
        ),
        verify_cfg=verify_cfg,
    )


@pytest.mark.parametrize("p", [0.1, 0.35, 0.8])
def test_dropout(test_kind, test_device, p):
    ones = torch.ones(1, 1, 1024, 1024, requires_grad=test_kind.is_training())
    x = Tensor.create_from_torch(ones)

    def validate_dropout(golden, result):
        tolerance = 0.05
        golden_sum = torch.sum(golden)
        result_sum = torch.sum(result)
        error = torch.abs((golden_sum - result_sum) / golden_sum).item()
        return error < tolerance

    @run(
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            relative_atol=100.0,
            pcc=-0.1,
            golden_compare_callback=validate_dropout),
    )
    def simple_dropout(x):
        return pybuda.op.Dropout("dropout0", x, p=p, training=test_kind.is_training())

    simple_dropout(x)


def test_matmul_gradient_t(test_kind, test_device):
    shape = (1, 3, 128, 128)

    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc),
    )
    def simple_matmul_gradient_t(x, weight=None):
        return pybuda.op.Matmul("mm0", x, weight)

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training()))
    w = pybuda.Parameter(*shape, requires_grad=test_kind.is_training())
    simple_matmul_gradient_t(x, weight=w)

class ComparisonTest(PyBudaModule):
    """
    Test wrapper for comparison operators
    """

    op_map = {
        "eq": pybuda.op.Equal,
        "ne": pybuda.op.NotEqual,
        "gt": pybuda.op.Greater,
        "lt": pybuda.op.Less,
        "ge": pybuda.op.GreaterEqual,
        "le": pybuda.op.LessEqual
    }

    def __init__(self, name, op_type):
        super().__init__(name)
        assert (op_type in ComparisonTest.op_map) and (ComparisonTest.op_map[op_type] is not None), f"Comparison operator, {op_type} is not defined. "
        self.op_type = op_type

    def forward(self, act1, act2):
        return ComparisonTest.op_map[self.op_type](self.op_type, act1, act2)


@pytest.mark.parametrize("op_type", ["eq", "ne", "gt", "lt", "ge", "le"])
@pytest.mark.parametrize("shape", ([1, 1, 64, 64], [128, 768], [1, 340, 180]), ids=["shape1x1x64x64", "shape128x768", "shape=1x340x180"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
def test_comparison(
    training,
    test_device, 
    recompute,
    shape,
    op_type
):

    verify_cfg.run_net2pipe=False #tenstorrent/pybuda#1078
    if training:
        pytest.skip("Comparison operators shouldn't have derivative, and backward.")

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    mod = ComparisonTest(name="test_comparison", op_type=op_type)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(mod)

    act1, act2 = torch.normal(mean=0, std=0.1, size=shape,), torch.normal(mean=1, std=0.1, size=shape,)
    act1.requires_grad = True
    act2.requires_grad = True
    act1 = Tensor.create_from_torch(act1)
    act2 = Tensor.create_from_torch(act2)

    pybuda_compile(
        tt0,
        "comparison",
        act1,
        act2,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=verify_cfg,
    )


class ClipTest(PyBudaModule):
    """
    Test wrapper for clip
    """

    def __init__(self, name, min_value, max_value):
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, act):
        return pybuda.op.Clip("clip", act, self.min_value, self.max_value)


@pytest.mark.parametrize("shape", ([128, 768], [1, 70, 90]), ids=["shape=128x768", "shape=1x1x70x90"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("max_value", (0.62, 1.0, -0.5), ids=["max=0.62", "max=1.0", "max=-0.5"])
@pytest.mark.parametrize("min_value", (0.32, -0.9, 0.0, -0.5), ids=["min=0.32", "min=-0.9", "min=0.0", "min=-0.5"])
def test_clip(
    training,
    test_device, 
    recompute, 
    min_value,
    max_value,
    shape
):
    if test_device.is_grayskull():
        verify_cfg.run_net2pipe=False #tenstorrent/pybuda#1078

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    mod = ClipTest(name="test_clip", min_value=min_value, max_value=max_value)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(mod)

    # tensor in range (-10.0, 10.0)
    act1 = torch.rand(shape) * 20.0 - 10.0
    act1.requires_grad=True
    act1 = Tensor.create_from_torch(act1)

    pybuda_compile(
        tt0,
        "clip",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=verify_cfg,
    )


class HeavisideTest(PyBudaModule):
    """
    Test wrapper for heaviside operator
    """

    def __init__(self, name):
        super().__init__(name)

    def forward(self, act1, act2):
        return pybuda.op.Heaviside("heaviside", act1, act2)


@pytest.mark.parametrize("shape", ([1, 1, 64, 64], [128, 768], [1, 340, 180]), ids=["shape1x1x64x64", "shape128x768", "shape=1x340x180"])
@pytest.mark.parametrize("recompute", (True, False), ids=["Recompute", "NoRecompute"])
def test_heaviside(
    training,
    test_device, 
    recompute,
    shape,
):
    verify_cfg.run_net2pipe=False #tenstorrent/pybuda#1078
    if training:
        pytest.skip("Heaviside shouldn't have derivative, and backward.")

    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    mod = HeavisideTest(name="test_heaviside")
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(shape, requires_grad=True))
    act2 = torch.rand(shape) * (1.0 * torch.randint(0, 2, shape))
    act2.requires_grad=True
    act2 = Tensor.create_from_torch(act2)


    pybuda_compile(
        tt0,
        "heaviside",
        act1,
        act2,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=verify_cfg,
    )

def test_matmul_relu(test_kind):
    def matmul_relu(act, *, weights):
        op0 = pybuda.op.Matmul(f"op0", act, weights)
        op1 = pybuda.op.Relu(f"op1", op0)
        return op1

    module = ModuleBuilder(matmul_relu, weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=test_kind))


def test_matmul_gelu_matmul(test_kind):
    def matmul_gelu(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Gelu(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2

    module = ModuleBuilder(matmul_gelu, ff1_weights=pybuda.Parameter(1,1,64,64), ff2_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=test_kind, optimizer=None))


def test_consumer_ops_belonging_to_different_epochs(test_kind):
    def consumer_ops_belonging_to_different_epochs(act, *, weights):
        op0 = pybuda.op.Matmul(f"op0", act, weights)
        op1 = pybuda.op.Buffer(f"buffer_a", op0)
        op2 = pybuda.op.Buffer(f"buffer_b", op1)
        op3 = pybuda.op.Buffer(f"buffer_c", op1)
        op3 = pybuda.op.Add(f"add", op2, op3)
        return op3
    
    pybuda.set_epoch_break("buffer_a")
    pybuda.set_epoch_break("buffer_b")
    pybuda.set_epoch_break("buffer_c")

    module = ModuleBuilder(consumer_ops_belonging_to_different_epochs, weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=test_kind))
    

def test_consumer_ops_belonging_to_different_chips(test_kind):
    """
    Let's suppose we have:
                         |     -> buffer_b (epoch1)
                         |   /
    buffer_a (epoch0) -> | -
                         |   \
                         |     -> buffer_c (epoch2)
                         
       chip0                    chip1
    
    There should only be a single e2e queue generated in this situation, rather than two.
    """
    def consumer_ops_belonging_to_different_chips(act, *, weights):
        op0 = pybuda.op.Matmul(f"op0", act, weights)
        op1 = pybuda.op.Buffer(f"buffer_a", op0)
        op2 = pybuda.op.Buffer(f"buffer_b", op1)
        op3 = pybuda.op.Buffer(f"buffer_c", op1)
        op3 = pybuda.op.Add(f"add", op2, op3)
        return op3
    
    pybuda.set_epoch_break("buffer_a")
    pybuda.set_chip_break("buffer_b")
    pybuda.set_epoch_break("buffer_c")

    arch = backend_devices[os.environ.get("BACKEND_ARCH_NAME", "grayskull")]
    compiler_cfg = _get_global_compiler_config()
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False if arch is BackendDevice.Grayskull else True

    module = ModuleBuilder(consumer_ops_belonging_to_different_chips, weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)],
            VerifyConfig(test_kind=test_kind, arch=arch, chip_ids=list(range(2))))


def test_matmul_buffer_matmul(test_kind):
    def matmul_buffer_matmul(act, *, ff1_weights, ff2_weights):
        op0 = pybuda.op.Matmul(f"ff1", act, ff1_weights)
        op1 = pybuda.op.Buffer(f"gelu", op0)
        op2 = pybuda.op.Matmul(f"ff2", op1, ff2_weights)
        return op2
    
    pybuda.set_epoch_break("gelu")
    pybuda.set_epoch_break("ff2")

    module = ModuleBuilder(matmul_buffer_matmul, ff1_weights=pybuda.Parameter(1,1,64,64), ff2_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)], VerifyConfig(test_kind=test_kind))


def test_z_sparse_matmul(test_device):
    input_shape = (1, 64, 128, 128)

    class Model(PyBudaModule):
        def __init__(self):
            super().__init__(name="sparsematmul_test")
            rows = torch.arange(0, 128).tolist()
            cols = rows
            sparse = torch.sparse_coo_tensor([rows, cols],torch.ones(len(cols)), (128, 128), dtype=torch.float32)
            sparse = torch.stack([sparse]*64, -3)
            sparse = torch.unsqueeze(sparse, 0) 
            self.add_constant("sparse")
            self.set_constant("sparse", pybuda.Tensor.create_from_torch(sparse, constant=True))

        def forward(self, x):
            out = pybuda.op.SparseMatmul("", self.get_constant("sparse"), x)
            return out

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    pybuda.verify.verify_module(
        Model(),
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )



class PowTest(pybuda.PyBudaModule):
    def __init__(self, name, exp_val):
        super().__init__(name) 
        self.exp_val = exp_val
            
    def forward(self, act1): 
        p1 = pybuda.op.Pow("pow", act1, self.exp_val)
        return p1


@pytest.mark.parametrize("is_exp_fp", [True, False])
def test_pow(test_device, test_kind, is_exp_fp):
    # set exponential value
    random.seed(1)
    if is_exp_fp:
        exp_val = random.random()
    else:
        exp_val = int(2) 

    x = Tensor.create_from_torch(torch.rand((1, 1, 64, 64), requires_grad=True))  
    pybuda.verify.verify_module(
        PowTest("pow-test", exp_val),
        ([1,1,64,64]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.99
        ),
        inputs = [(x)],
    )


class PowerBinaryTest(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name) 
           
    def forward(self, act1, act2): 
        p1 = pybuda.op.Power("power-binary", act1, act2)
        return p1


def test_power_binary(test_device, test_kind):
    x = Tensor.create_from_torch(torch.rand((1, 1, 64, 64), requires_grad=True)) 
    y = Tensor.create_from_torch(torch.rand((1, 1, 64, 64), requires_grad=True)) 
    pybuda.verify.verify_module(
        PowerBinaryTest("power-binary-test"),
        ([1,1,64,64]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.99
        ),
        inputs = [(x,y)],
    )


class ReluTest(pybuda.PyBudaModule):
    def __init__(self, name, _threshold, _mode):
        super().__init__(name) 
        self.threshold = _threshold
        self.mode = _mode
           
    def forward(self, act1): 
        p1 = pybuda.op.Relu("relu", act1, self.threshold, self.mode)
        return p1


@pytest.mark.parametrize("threshold", [0.0, 0.1], ids=["default", "custom"])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_relu(test_device, test_kind, threshold, mode):
    if threshold == 0.0 and mode == "max":
        pytest.skip("inv-relu is not supposed to be called with the default threshold")    
 
    x = Tensor.create_from_torch(torch.randn((1, 1, 64, 64), requires_grad=True))    
    pybuda.verify.verify_module(
        ReluTest("relu-test", threshold, mode),
        ([1,1,64,64]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind, 
        ),
        inputs = [(x)],
    )


class BinaryTest(pybuda.PyBudaModule):
    def __init__(self, name, _mode):
        super().__init__(name)  
        self.mode = _mode
           
    def forward(self, act1, act2): 
        if self.mode == "less":
            p1 = pybuda.op.Less("less", act1, act2)
        elif self.mode == "gteq":
            p1 = pybuda.op.GreaterEqual("gteq", act1, act2)       
        elif self.mode == "heaviside": 
            p1 = pybuda.op.Heaviside("heaviside", act1, act2)   
        elif self.mode == "lteq":
            p1 = pybuda.op.LessEqual("lteq", act1, act2)       
        elif self.mode == "greater":
            p1 = pybuda.op.Greater("greater", act1, act2)   
        elif self.mode == "ne":
            p1 = pybuda.op.NotEqual("ne", act1, act2)       
        else:
            p1 = pybuda.op.Equal("eq", act1, act2)   
        return p1


@pytest.mark.parametrize("mode", ["less", "greater", "lteq", "gteq", "ne", "eq", "heaviside"])
def test_binary(test_device, mode):
    x = Tensor.create_from_torch(torch.randn((1, 1, 64, 64), requires_grad=True)) 
    y = Tensor.create_from_torch(torch.randn((1, 1, 64, 64), requires_grad=True)) 
    pybuda.verify.verify_module(
        BinaryTest(f"binary-{mode}-test", mode),
        ([1,1,64,64]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
        inputs = [(x,y)],
    )


@pytest.mark.parametrize("shape", [(128, 32, 256), (128, 8, 256), (128, 16, 256), (128, 16, 16)])
def test_large_reshape(shape):
    outer = shape[0]         
    num_blocks = shape[1]     
    block_size = shape[2] 
    
    @compile(
        compiler_cfg=CompilerConfig(enable_training=False, compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER),
        verify_cfg=VerifyConfig(run_golden=True),  # reshape not supported by backend
    )
    def simple_large_reshape(x, y): 
        x = pybuda.op.Multiply("mult0", x, x) 
        x = pybuda.op.Reshape("reshape0", x, (1,outer,num_blocks,block_size))  
        y = pybuda.op.Multiply("mult1", x, y) 
        return y
 
    x = Tensor.create_from_torch(torch.rand((outer, num_blocks*block_size))) 
    y = Tensor.create_from_torch(torch.rand((1,outer,num_blocks,block_size)))
    simple_large_reshape(x,y)


def test_invalid_vstack_candidate(test_kind, test_device):
    class Model(pybuda.PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            self.add_constant("c")
            self.set_constant("c", torch.ones((1, 256, 1, 1)))
            self.add_parameter("b", pybuda.Parameter(*(324,), requires_grad=True))
            self.add_parameter("w", pybuda.Parameter(*(324, 256, 3, 3), requires_grad=True))
        
        def forward(self, x):
            x = pybuda.op.Add("", x, self.get_constant("c"))
            x = pybuda.op.Conv2d("", x, self.get_parameter("w"), self.get_parameter("b"), 1, (1, 1, 1, 1), 1, 1, 0)
            return x
    
    module = Model("invalid_vstack_candidate")
    verify_module(
        module,
        ((1, 256, 1, 1),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

def test_intermediate_verification(test_kind):
    if test_kind.is_training():
        pytest.skip()

    class InterVer(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            const0 = torch.ones((1,1))
            self.add_constant("const0")
            self.set_constant("const0", pybuda.Tensor.create_from_torch(const0, constant=True))
    
            const1 = torch.ones((1,1))
            self.add_constant("const1")
            self.set_constant("const1", pybuda.Tensor.create_from_torch(const1, constant=True))
            const2 = torch.ones((1,1,8,1))
            self.add_constant("const2")
            self.set_constant("const2", pybuda.Tensor.create_from_torch(const2, constant=True))
    
        def forward(self, inp):
            index = pybuda.op.Index("index", inp, -1, 0, 512, 1)
            reshape0 = pybuda.op.Reshape("Reshape0", index, [1, 64, 8, 64])
            mult = pybuda.op.Multiply("Mul", reshape0, reshape0)
            reduce_sum = pybuda.op.ReduceSum("Sum", mult, -1)
            reshape1 = pybuda.op.Reshape("Reshape1", reduce_sum, [1, 64, 8, 1])
            sqrt = pybuda.op.Sqrt("Sqrt", reshape1)
            max = pybuda.op.Max("max", sqrt, self.get_constant("const0"))
            min = pybuda.op.Min("min", max, self.get_constant("const1"))
            sub = pybuda.op.Subtract("Sub", sqrt, min,)
            add = pybuda.op.Add("Add", sqrt, sub)

            recip = pybuda.op.Reciprocal("Recip", add)
            mult2 = pybuda.op.Multiply("mul2", recip, reshape0)
            mult3 = pybuda.op.Multiply("mul3", mult2, self.get_constant("const2"))
            trans = pybuda.op.Transpose("transpose", mult3, -3, -2, 8)
            return trans

    compiler_config = _get_global_compiler_config()
    mod = InterVer("Intermediate_verification")
    verify_module(mod, [(1, 64, 1024)], VerifyConfig(test_kind=test_kind, verify_all=True))

def test_channel_fuse_concat_select(test_kind):
    if test_kind.is_training():
        pytest.skip()

    class channel_select_fusion(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            const0 = torch.ones((1,1))
            self.add_constant("const0")
            self.set_constant("const0", pybuda.Tensor.create_from_torch(const0, constant=True))
    
            const1 = torch.ones((1,1))
            self.add_constant("const1")
            self.set_constant("const1", pybuda.Tensor.create_from_torch(const1, constant=True))
            const2 = torch.ones((1,1))
            self.add_constant("const2")
            self.set_constant("const2", pybuda.Tensor.create_from_torch(const2, constant=True))

    
        def forward(self, inp):
            index0 = pybuda.op.Index("index0", inp, -3, 0, 1, 1)
            index1 = pybuda.op.Index("index1", inp, -3, 1, 2, 1)
            index2 = pybuda.op.Index("index2", inp, -3, 2, 3, 1)

            mult0 = pybuda.op.Multiply("Mul0", index0, self.get_constant("const0"))
            mult1 = pybuda.op.Multiply("Mul1", index1, self.get_constant("const1"))
            add2 = pybuda.op.Add("Add2", mult1, self.get_constant("const2"))
            concat = pybuda.op.Concatenate("Concat", mult0, add2, index2, axis=-3)
            m1 = pybuda.op.Matmul("matmul1", concat, concat)

            return m1

    compiler_config = _get_global_compiler_config()
    mod = channel_select_fusion("channel_select_fusion")
    verify_module(mod, [(1, 3, 224, 224)], VerifyConfig(test_kind=test_kind, verify_all=True))

def test_erase_consecutive_reshape_binary(test_kind): 
    inp_shape = (1, 1, 1472, 16)
    inter_shape = (1, 16, 32, 46)
    param_shape = (1, 16, 1, 1)
    out_shape = (1, 1, 4, 5888)

    class Model(PyBudaModule):
        def __init__(self, name, inter_shape, out_shape, param_shape):
            super().__init__(name)
            self.inter_shape = inter_shape
            self.out_shape = out_shape
            self.param_shape = param_shape
            self.param = pybuda.Parameter(*self.param_shape, requires_grad=True)

        def forward(self, x, y):
            x = pybuda.op.Multiply("mult0", x, x)
            x = pybuda.op.Transpose("t0", x, -2, -1)
            x = pybuda.op.Reshape("reshape0", x, self.inter_shape)
            x = pybuda.op.Add("add0", x, self.param)
            x = pybuda.op.Reshape("reshape1", x, self.out_shape)
            x = pybuda.op.Multiply("multiply", x, y)
            return x

    mod = Model("consecutive_reshape_binary", inter_shape, out_shape, param_shape)
    verify_module(
        mod,
        (inp_shape, out_shape),
        verify_cfg=VerifyConfig( 
            test_kind=test_kind,
        ),
    )

def test_dual_reduce(test_kind):
    input_shape = (1, 1, 3, 1024)

    class Model(PyBudaModule):
        def __init__(self):
            super().__init__("dual_reduce")

        def forward(self, x):
            x = pybuda.op.Softmax("", x, dim=-1)
            x = pybuda.op.Reshape("", x, (1, 3, 32, 32))
            x = pybuda.op.ReduceSum("", x, dim=-2)
            x = pybuda.op.ReduceSum("", x, dim=-1)
            x = pybuda.op.Reshape("", x, (1, 1, 3, 1))
            x = pybuda.op.Softmax("", x, dim=-1)
            return x

    mod = Model()
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig( 
            test_kind=test_kind,
        ),
    )

@pytest.mark.parametrize("seq_len", [1, 32, 64, 112, 192])
@pytest.mark.parametrize("grid_r", [1, 2, 3])
@pytest.mark.parametrize("grid_c", [1, 2])
def test_embedding(test_device, seq_len, grid_r, grid_c):
    if (align_up_tile(seq_len) // 32) % grid_r != 0:
        pytest.skip()

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_embedding(x, table=None):
        x = pybuda.op.Embedding("embedding", table, x)
        return x

    compiler_config = _get_global_compiler_config()
    compiler_config.enable_tvm_cpu_fallback = False

    pybuda.config.override_op_size("embedding", (grid_r, grid_c))

    dictionary_size = 64
    hidden_dim = 128
    x = Tensor.create_from_torch(torch.randint(dictionary_size, (1, seq_len), dtype=torch.int))
    table = pybuda.Parameter.create_from_torch(torch.nn.Parameter(torch.randn((dictionary_size, hidden_dim))))
    simple_embedding(x, table=table)

@pytest.mark.parametrize("input_shape", [(1, 10, 32, 32), (1, 32, 16, 16),])
@pytest.mark.parametrize("axis", [-3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("num_operands", [2,3])
def test_interleave(test_kind, test_device, input_shape, axis, stride, num_operands):
    class Model(PyBudaModule):
        def __init__(self, name, axis, stride):
            super().__init__(name)
            self.axis = axis
            self.stride = stride

        def forward(self, *operands):
            x = pybuda.op.Interleave("interleave0", *operands, axis=self.axis, stride=self.stride)
            return x

    input_shapes = tuple([input_shape for _ in range(num_operands)])
    mod = Model("interleave_test", axis, stride)
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )

@pytest.mark.parametrize("mode", ["hslice", "hstack", "vslice", "vstack"])
def test_slice_stack_non_tile_aligned(test_kind, test_device, mode):
    class SliceStackModule(PyBudaModule):
        def __init__(self, name, factor, mode):
            super().__init__(name)
            self.factor = factor
            self.mode = mode

        def forward(self, activations):
            if mode == "hslice":
                ret = pybuda.op.HSlice("hslice0", activations, self.factor)
            elif mode == "hstack":
                ret = pybuda.op.HStack("hstack0", activations, self.factor)
            elif mode == "vslice":
                ret = pybuda.op.VSlice("vslice0", activations, self.factor)
            else:
                ret = pybuda.op.VStack("vstack0", activations, self.factor)
            return ret

    # input shape
    input_shape = ()
    if mode == "hslice":
        input_shape = (1,3,1,9)
    elif mode == "vslice":
        input_shape = (1,3,9,1)
    else:
        input_shape = (1,9,1,1)

    mod = SliceStackModule("test_slice_stack_not_aligned", 3, mode)
    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        mod,
        [input_shape],
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            relative_atol=relative_atol,
            pcc=pcc
        )
    )

def test_concat_two_kinds_pad(test_device):
    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            self.add_parameter("w", pybuda.Parameter(*(1, 1, 352, 192), requires_grad=True))

        def forward(self, in0, in1, in2, in3, in4, in5, y):
            in0 = pybuda.op.Multiply("m0", in0, in0)
            in1 = pybuda.op.Multiply("m1", in1, in2)
            in2 = pybuda.op.Multiply("m2", in2, in3)
            in3 = pybuda.op.Multiply("m3", in3, in4)
            in4 = pybuda.op.Multiply("m4", in4, in4)
            in5 = pybuda.op.Multiply("m5", in5, in1)
            x = pybuda.op.Concatenate("", in0, in1, in2, in3, in4, in5, axis=-1)
            x = pybuda.op.Multiply("m6", x, y)
            x = pybuda.op.PadTile("p0", x, -1, 336)
            x = pybuda.op.Matmul("mm0", x, self.get_parameter("w"))
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    # compiler_cfg.place_on_new_epoch("m6_transpose_nop_0")
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"
    os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    # input shape
    common_len = 3136
    input_shapes = ((1, 1, common_len, 96), (1, 1, common_len, 48), (1, 1, common_len, 48), (1, 1, common_len, 48), (1, 1, common_len, 48), (1, 1, common_len, 48), (1, 1, common_len, 336))
    mod = Module("test_concat_two_kinds_pad")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )

    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{}"


def test_negative_reduce_max(test_device):
    df = pybuda.config.DataFormat.Float16
    pybuda.config.set_configuration_options(default_df_override=df, accumulate_df=df)

    def f(a, b):
        mae = torch.mean(torch.abs(a - b))
        assert mae.item() < 0.1
        return mae.item() < 0.1

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            golden_compare_callback=f),
    )
    def negative_reduce_max(a):
        return pybuda.op.ReduceMax("reduce", a, dim=-1)

    a = Tensor.create_from_torch(torch.randn(1, 1, 32, 32) - 100.0)
    negative_reduce_max(a)

@pytest.mark.parametrize("dims", [(1,1,128,128)])
def test_unary_transpose(test_device, dims):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def eltwise_unary_transpose(x):
        opA = pybuda.op.Transpose(name='',operandA=x, dim0=-2, dim1=-1)
        return pybuda.op.Exp('', opA)
    
    compiler_config = _get_global_compiler_config()
    compiler_config.enable_tvm_cpu_fallback = False

    originalX = Tensor.create_from_torch(torch.randn(dims)) # a x b
    eltwise_unary_transpose(originalX)

@pytest.mark.parametrize("dims", [(1,1,128,64)])
@pytest.mark.parametrize("trans_both",[True, False])
def test_binary_transpose(test_device, dims, trans_both):
    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def eltwise_binary_transpose(x, y):
        opA = pybuda.op.Transpose(name='transA',operandA=x, dim0=-2, dim1=-1) if trans_both else x #axb
        opB = pybuda.op.Transpose(name='transB',operandA=y, dim0=-2, dim1=-1) 
        return pybuda.op.Add('', opA, opB)

    compiler_config = _get_global_compiler_config()
    compiler_config.enable_tvm_cpu_fallback = False

    originalX = Tensor.create_from_torch(torch.randn(dims)) # a x b
    originalY = torch.randn(dims) if trans_both else torch.randn(dims).transpose(-2,-1)
    originalY = Tensor.create_from_torch(originalY) # # a x b

    print(f"originalX: {originalX.shape}")
    print(f"originalY: {originalY.shape}")
    eltwise_binary_transpose(originalX, originalY)


def test_grad_eltwise_op(test_device):
    """
    x   W
    |  / |
    mul  |
     \   |
       op
    """
    shape = (1, 1, 512, 512)
    test_kind = TestKind.TRAINING

    @run(
        verify_cfg=VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def forked_op(x, weight=None):
        prod = pybuda.op.Matmul("", x, weight)
        op = pybuda.op.Add("", prod, weight)
        return op
    
    compiler_config = _get_global_compiler_config()
    compiler_config.enable_tvm_cpu_fallback = False

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training(), dtype=torch.bfloat16))
    w = pybuda.Parameter(torch.randn(shape, requires_grad=test_kind.is_training(), dtype=torch.bfloat16))

    forked_op(x, weight=w)


def test_3d_mm(test_device):
    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            self.add_parameter("param", pybuda.Parameter(*(1, 1), requires_grad=True))

        def forward(self, x): 
            y = pybuda.op.Multiply("", x, self.get_parameter("param"))
            y = pybuda.op.HSlice("", y, 8)

            x = pybuda.op.HSlice("", x, 8)
            x = pybuda.op.Transpose("", x, 2, 3) 
            x = pybuda.op.Matmul("", y, x)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    #compiler_cfg.balancer_op_override("multiply_0", "t_stream_shape", (1,1))
    #compiler_cfg.balancer_op_override("matmul_4", "t_stream_shape", (2,1))
    #compiler_cfg.balancer_op_override("matmul_4", "t_stream_dir", "r")
 
    input_shapes = ((1, 1, 256, 512),)

    # input shape 
    mod = Module("test_3d_mm")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )
    

def test_multipliers_overrides(test_device):
    shape = (1, 1, 32, 32)
    test_kind = TestKind.INFERENCE

    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def simple_matmul_buffer_overrides(x, weight=None):
        return pybuda.op.Matmul("mm0", x, weight)

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training()))
    w = pybuda.Parameter(torch.randn(shape, requires_grad=test_kind.is_training()))
    pybuda.config.override_input_buffer_multiplier("mm0", 0, multiplier=4)
    pybuda.config.internal_override_output_buffer_multiplier("mm0", multiplier=4)

    simple_matmul_buffer_overrides(x, weight=w)


def test_broadcast_transpose(test_device):

    @run(test_device)
    def broadcast_transpose(x):
        x = pybuda.op.Broadcast("", x, -2, 64)
        return pybuda.op.Transpose("", x, -2, -1)

    broadcast_transpose(Tensor.create_from_torch(torch.randn(1, 1, 1, 128)))


def test_scalar_matmul_bias(test_device):
    pybuda.set_configuration_options(backend_output_dir=f"tt_build/test_scalar_matmul_bias")
    @run(test_device)
    def scalar_matmul_bias(a, w=None, b=None):
        x = pybuda.op.Matmul("", a, w)
        x = pybuda.op.Add("", x, b)
        return x
    
    x = Tensor.create_from_torch(torch.randn(1, 1, 32, 32))
    w = pybuda.Parameter.create_from_torch(torch.randn(1, 1, 32, 128))
    tmp = torch.zeros(1, 1, 1, 1)
    tmp[0, 0, 0, 0] = 1000.0
    b = pybuda.Parameter.create_from_torch(tmp)
    scalar_matmul_bias(x, w=w, b=b)


def test_mismatch_repro(test_device):
    pytest.xfail()

    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name) 
            self.add_parameter("features.7.weight", pybuda.Parameter(*(64, 32, 3, 3), requires_grad=True))
            self.add_parameter("features.7.bias", pybuda.Parameter(*(64,), requires_grad=True))

        def forward(self, x): 
            x = pybuda.op.Conv2d("", x, self.get_parameter("features.7.weight"), self.get_parameter("features.7.bias"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.balancer_op_override("conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (2,1))
    import os 
    os.environ["PYBUDA_REPRODUCE_SUBGRAPH"]  = "1"
    os.environ["PYBUDA_REPRODUCE_SUBGRAPH_INPUT"] = "conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2"
    os.environ["PYBUDA_REPRODUCE_SUBGRAPH_OUTPUT"] = "conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2"
 
    input_shapes = ((1, 32, 16, 16),)

    # input shape 
    mod = Module("test_mismatch_repro")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )


def test_mismatch_repro_smm(test_device):
    pytest.xfail()

    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)
            idx = torch.arange(256).tolist()
            sparse = torch.sparse_coo_tensor([idx, idx],torch.ones(256), (256, 256))
            sparse = torch.stack([sparse]*9, -3)
            sparse = torch.unsqueeze(sparse, 0)
            self.add_constant("sparse")
            self.set_constant("sparse", pybuda.Tensor.create_from_torch(sparse, constant=True))

        def forward(self, x):
            x = pybuda.op.Transpose("", x, -1, -2)
            x = pybuda.op.SparseMatmul("", self.get_constant("sparse"), x)
            x = pybuda.op.VStack("", x, 9)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.balancer_op_override("sparse_matmul_1.dc.sparse_matmul.1.lc2", "t_stream_shape", (2,1))
    compiler_cfg.balancer_op_override("sparse_matmul_1.dc.sparse_matmul.1.lc2", "grid_shape", (2,1))

    input_shapes = ((1, 1, 32, 256),)

    # input shape
    mod = Module("test_mismatch_repro_smm")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )


def test_multi_repeat(test_device):
    @run(test_device)
    def multi_repeat(x):
        x = pybuda.op.Repeat("", x, [1, 1, 1, 2]);
        x = pybuda.op.Repeat("", x, [1, 1, 1, 2]);
        return x

    x = Tensor.create_from_torch(torch.rand(1, 1, 32, 32))
    multi_repeat(x)


def get_device_intermediates(op_intermediates: List[str]) -> Dict[str, List[torch.Tensor]]:
    device_intermediates: Dict[str, List[torch.Tensor]] = defaultdict(list)
    intermediates_queue = pybuda.get_intermediates_queue()

    while not intermediates_queue.empty():
        intermediate_tensors = intermediates_queue.get()
        for name, intermediate_tensor in zip(op_intermediates, intermediate_tensors):
            device_intermediates[name].append(intermediate_tensor.to_pytorch())
    return device_intermediates


def test_read_back_intermediates(test_kind, test_device):
    if test_kind.is_training():
        op_intermediates = ["matmul_intermediate", "bw_in0_matmul_output_matmul_1"]
    else:
        op_intermediates = ["matmul_intermediate"]

    pybuda.set_configuration_options(op_intermediates_to_save=op_intermediates)
    num_inputs = 4

    @run(
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            intermediates=True,
            microbatch_count=num_inputs,
        ),
        num_inputs=num_inputs,
    )
    def fetch_intermediates(x0, x1, x2):
        intermediate = pybuda.op.Matmul("matmul_intermediate", x0, x1)
        return pybuda.op.Matmul("matmul_output", intermediate, x2)

    x = Tensor.create_from_torch(torch.randn(1, 1, 63, 63, requires_grad=test_kind.is_training()))
    y = Tensor.create_from_torch(torch.randn(1, 1, 63, 63, requires_grad=test_kind.is_training()))
    z = Tensor.create_from_torch(torch.randn(1, 1, 63, 63, requires_grad=test_kind.is_training()))
    fetch_intermediates(x, y, z)

    device = pybuda.get_tenstorrent_device()
    compiled_results = device.get_compiled_results()

    golden_intermediates: Dict[str, torch.Tensor ] = compiled_results.golden_intermediates  # golden replicated
    device_intermediates: Dict[str, List[torch.Tensor]] = get_device_intermediates(op_intermediates)

    for op_name in op_intermediates:
        assert (len(device_intermediates[op_name]) == num_inputs), f"Expected {num_inputs} intermediate tensors for {op_name}"
        if op_name in golden_intermediates:
            for idx in range(num_inputs):
                compare_tensor_to_golden(
                    op_name,
                    golden_intermediates[op_name],
                    device_intermediates[op_name][idx],
                    is_buda=True,
                )


def test_2d_daisy_chain(test_device):
    @run(test_device)
    def daisy_chain_2d(x):
        rows, columns = 16, 4
        outputs = []
        for i in range(rows):
            for j in range(columns):
                op = pybuda.op.Gelu(f"gelu_{i}_{j}", x)
                outputs.append(op)

        input = "inputs"

        # insert daisy-chain along each column
        for j in range(columns):
            gelu_rows = [f"gelu_{i}_{j}" for i in range(rows)]
            pybuda.insert_nop(input, gelu_rows, daisy_chain=True)

        # insert daisy-chain across first row
        gelu_first_row = [f"buffer_0_inputs_gelu_{0}_{j}" for j in range(columns)]
        pybuda.insert_nop(input, gelu_first_row, daisy_chain=True)
        
        return outputs

    x = Tensor.create_from_torch(torch.rand(1, 1, 32, 32))
    daisy_chain_2d(x)


def test_forked_dram_inputs(test_device):
    @run(test_device)
    def forked_dram_inputs(x):
        op_gelu1 = pybuda.op.Gelu(f"", x)
        op_gelu2 = pybuda.op.Gelu(f"", x)
        op_output = pybuda.op.Add(f"", op_gelu1, op_gelu2)
        return op_output
    pybuda.config.set_configuration_options(enable_auto_fusing=False)
    x = Tensor.create_from_torch(torch.rand(1, 1, 32, 32))
    forked_dram_inputs(x)


def test_conv3d(test_device):
    inC, inD, inH, inW = (2, 5, 5, 5)
    outC, kD, kH, kW = (4, 3, 3, 3)
    stride = 1 
    padding = 0
    dilation = 1 

    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name) 
            self.add_parameter("weight", pybuda.Parameter(*(outC, inC, kD, kH, kW), requires_grad=True)) 

        def forward(self, x): 
            x = pybuda.op.Conv3d("", x, self.get_parameter("weight"), None, stride=[stride, stride, stride], padding=[padding, padding, padding, padding, padding, padding], dilation=dilation, groups=1, channel_last=0)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
 
    input_shapes = ((1, inC, inD, inH, inW),)

    # input shape 
    mod = Module("test_conv3d")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )

def test_maxpool3d(test_device): 
    inC, inD, inH, inW = (3, 8, 8, 8)
    outC, kD, kH, kW = (3, 3, 3, 3)
    assert inC == outC
    stride = 1 
    padding = 0
    dilation = 1 

    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name) 

        def forward(self, x): 
            x = pybuda.op.MaxPool3d("", x, (kD, kH, kW), stride=stride, padding=padding, dilation=dilation, channel_last=0)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
 
    input_shapes = ((1, inC, inD, inH, inW),)

    # input shape 
    mod = Module("test_maxpool3d")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )

def test_resize3d(test_device):
    inD, inH, inW = (8, 32, 32)
    outD, outH, outW = (16, 64, 64)

    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x):
            x = pybuda.op.Resize3d("", x, (outD, outH, outW), channel_last=0)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"

    input_shapes = ((1, 3, inD, inH, inW),)

    # input shape
    mod = Module("test_resize3d")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )

def test_emulate_harvested(test_device):
    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, x):
            x = pybuda.op.Add("", x, x)
            return x

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object

    input_shapes = ((1, 3, 32, 32),)

    # input shape
    mod = Module("test")
    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        )
    )

def test_blackhole_golden_sanity():
    class Module(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, a, b, c):
            x = pybuda.op.Add("add0", a, b)
            x = pybuda.op.Matmul("matmul0", x, c)
            return x

    input_shapes = ((1, 3, 64, 64),(1, 3, 64, 64), (1, 3, 64, 64))

    # input shape
    module = Module("test_blackhole_golden_sanity")
    verify_module(
        module,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=BackendType.Golden,
            arch=BackendDevice.Blackhole,
        )
    )