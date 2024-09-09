# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch

from transformers import BertModel, BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

import pybuda
from pybuda import (
    PyTorchModule,
    TTDevice,
    PyBudaModule,
    BackendType,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
    Tensor,
    pybuda_compile,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

import torch
from torch import nn

@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_tvm_bert(training, recompute):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    config = BertConfig()

    config.num_hidden_layers = 2
    hidden_size = config.num_hidden_layers * 32

    config.hidden_size = hidden_size
    config.num_attention_heads = config.num_hidden_layers
    config.intermediate_size = hidden_size
    model = BertEncoder(config)

    shape = (1, hidden_size, hidden_size)
    hidden_states = torch.rand(shape)

    mod = PyTorchModule("bert_layer", model)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "bert_layer",
        hidden_states,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            # enable_consteval=False,
            enable_tvm_constant_prop=True,
            match_subgraph_patterns=config.num_hidden_layers,
            compile_depth=CompileDepth.POST_PATTERN_MATCHER,
        ),
        verify_cfg=VerifyConfig(
            intermediates=False,
        ),
    )
    match_result = ret.pass_specific_output_kwargs["match_result"]
    assert match_result.is_subgraph_pattern_found
    assert match_result.is_subgraph_loopable


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_linear_looped(training, recompute):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class BudaTest(PyBudaModule):
        shape = (1, 1, 64, 64)

        def __init__(self, name):
            super().__init__(name)
            self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
            self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

        def forward(self, act1):
            m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
            m1g = pybuda.op.Gelu("gelu1", m1)

            m2 = pybuda.op.Matmul("matmul2", m1g, self.weights2)
            m2g = pybuda.op.Gelu("gelu2", m2)

            return m2g

    mod = BudaTest("test_module")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape))

    mod.set_parameter("weights1", torch.rand(*BudaTest.shape, requires_grad=True))
    mod.set_parameter("weights2", torch.rand(*BudaTest.shape, requires_grad=True))

    pybuda_compile(
        tt0,
        "sanity",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            match_subgraph_patterns=2,
            compile_depth=CompileDepth.POST_PATTERN_MATCHER,
        ),
        verify_cfg=VerifyConfig(
            intermediates=False,
        ),
    )

def test_swin_roll():
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    class swin_roll(nn.Module):
        def __init__(self,shift_size):
            super().__init__()
            self.shift_size=shift_size
        def forward(self,hidden_state):
            shifted_hidden_state = torch.roll(hidden_state, shifts=(self.shift_size,self.shift_size), dims=(1,2))
            return shifted_hidden_state

    input_shape = (1,4,4,3)
    model = swin_roll(shift_size=-1)
    tt_model = PyTorchModule("pt_swin_roll", model)

    verify_module(
        tt_model,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=pybuda.BackendDevice.Wormhole_B0,
            devtype=pybuda.BackendType.Golden,
            test_kind=TestKind.INFERENCE,
        )
    )

@pytest.mark.parametrize("tranpose_dims", ((2, 0), (0, 1), (1, 2), (3, 1), (4, 1),(-1, -6)))
def test_reshape_transpose_reshape_tvm(test_device, tranpose_dims):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.compile_depth=CompileDepth.GENERATE_INITIAL_GRAPH
    class Model(nn.Module):
        def __init__(self, new_shape_1, dim0, dim1, new_shape_2):
            super().__init__()
            self.new_shape_1 = new_shape_1
            self.dim0 = dim0
            self.dim1 = dim1
            self.new_shape_2 = new_shape_2

        def forward(self,input):
            input = torch.reshape(input, self.new_shape_1)
            input = torch.transpose(input, self.dim0, self.dim1)
            input = torch.reshape(input, self.new_shape_2)
            return input

    new_shape_1 = (1, 4, 1, 1, 4, 9)
    dim0, dim1 = tranpose_dims
    new_shape_2 = (16, 9)

    input_shape = (1, 16, 9)
    model = Model(new_shape_1, dim0, dim1, new_shape_2)
    tt_model = PyTorchModule("pt_reshape_transpose_reshape", model)

    verify_module(
        tt_model,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_tvm_compile=True,
        )
    )

@pytest.mark.parametrize("input_shape", ((1, 4), (1, 4, 3), (1, 2, 7, 6)))
@pytest.mark.parametrize("repeat_dims", (1, 2, 3, -1, -2, -3))
@pytest.mark.parametrize("num_repeats", (2, 3))
def test_repeat_interleave_pytorch(test_device, input_shape, repeat_dims, num_repeats):

    dims = repeat_dims
    if dims < 0:
        dims = len(input_shape) + dims
        if dims < 0:
            pytest.skip()

    if dims > int(len(input_shape) - 1) or input_shape[dims] == 1:
        pytest.skip()

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    class Repeat_interleave_model(torch.nn.Module):
        def __init__(self, repeats, dims):
            super().__init__()
            self.repeats = repeats
            self.dims = dims

        def forward(self, input_tensor):
            return torch.repeat_interleave(input_tensor, repeats = self.repeats, dim = self.dims)

    model = Repeat_interleave_model(repeats=num_repeats, dims=repeat_dims)
    model.eval()

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(
        "pt_repeat_interleave", model
    )

    input_sample = torch.rand(input_shape)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(input_sample.shape,)],
        inputs=[(input_sample,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework=True,
            verify_tvm_compile=True,
        ),
    )


@pytest.mark.parametrize("alpha_val", (1.0, 0.367, 4.675, -8.0, -0.6743, 16.296))
def test_elu_pytorch(test_device, alpha_val):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    class elu_model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.elu = torch.nn.ELU(alpha=alpha_val)
        def forward(self, input_tensor):
            return self.elu(input_tensor)

    model = elu_model()
    model.eval()

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(
        "pt_elu", model
    )

    input_sample = torch.randint(-200, 200, (1, 3, 512, 512)).to(torch.float32)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(input_sample.shape,)],
        inputs=[(input_sample,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework=True,
            verify_tvm_compile=True,
        ),
    )
