# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from matplotlib import use
from pybuda.op.eval.common import compare_tensor_to_golden
from pybuda.verify.backend import verify_module
from test.tvm.recommendation.pytorch.deepctr_torch.layers.core_modules import LocalActivationUnit
from deepctr_torch.layers.core_modules import DNN, Conv2dSame, PredictionLayer
import torch
import numpy as np
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda
import pytest

import torch

from pybuda.config import CompileDepth, _get_global_compiler_config


inputs_dims = [64, 128]
hidden_units = [(32, 64), (16, 32, 64)]
activations = ['relu', 'sigmoid', 'linear', 'dice', 'prelu']

# NOTE: Operator repeat is not supported, all tests will fail
# TODO: When microbatch > 1 is supported in backend convert this to a verify_module test
# @pytest.mark.parametrize("activation", activations, ids=[s for s in activations])
# @pytest.mark.parametrize("hidden_units", hidden_units, ids=[f"hidden{str(s[0])}{str(s[1])}" for s in hidden_units])
# @pytest.mark.parametrize("embedding_dim", inputs_dims, ids=[f"dim{str(s)}" for s in inputs_dims])
# def test_local_activation_unit(training, embedding_dim, hidden_units, activation):

#     pytest.skip("Operator repeat is not supported")

#     recompute = True

#     if activation == 'prelu':
#         pytest.skip("nn.prelu not supported")

#     compile_depth = CompileDepth.POST_PATTERN_MATCHER

#     l2_reg = np.random.random()
#     dropout_rate = np.random.random()*0.5

#     class LAUWrapper(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = LocalActivationUnit(hidden_units=hidden_units, embedding_dim=embedding_dim, activation=activation, dropout_rate=dropout_rate,
#                     l2_reg=l2_reg, use_bn=False, dice_dim=2)
#         def forward(self, a, b):
#             return self.model(a, b)
    
#     model = LAUWrapper()
#     shape1 = (32, 1, embedding_dim)
#     shape2 = (32, 32, embedding_dim)

#     mod = PyTorchModule("deepctr_local_activation_unit", model)
    
#     sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
#     tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
#     tt0.place_module(mod)

#     act1 = torch.rand(*shape1)
#     act2 = torch.rand(*shape2)

#     ret = pybuda_compile(
#         tt0,
#         "deepctr_local_activation_unit",
#         act1,
#         act2,
#         compiler_cfg=CompilerConfig(
#             enable_training=training,
#             enable_recompute=recompute,
#             compile_depth=compile_depth
#         ),
#         verify_cfg=VerifyConfig(
#             intermediates=True,
#         ),
#     )

#     pytorch_out = model(act1, act2)
#     assert compare_tensor_to_golden("output", ret.golden_outputs[0], pytorch_out, is_buda=True, verify_cfg=VerifyConfig())


# TODO: When microbatch > 1 is supported in backend convert this to a verify_module test
@pytest.mark.parametrize("use_bn", [True, False], ids=["bn", "no_bn"])
@pytest.mark.parametrize("activation", activations, ids=[s for s in activations])
@pytest.mark.parametrize("hidden_units", hidden_units, ids=[f"hidden{str(s[0])}{str(s[1])}" for s in hidden_units])
@pytest.mark.parametrize("inputs_dim", inputs_dims, ids=[f"dim{str(s)}" for s in inputs_dims])
def test_dnn(test_kind, test_device, inputs_dim, hidden_units, activation, use_bn):

    if activation == 'prelu':
        pytest.skip("nn.prelu not supported")

    if activation == 'relu':
        pytest.skip("relu not supported in backend")

    compile_depth = CompileDepth.FULL

    l2_reg = np.random.random()
    dropout_rate = np.random.random()*0.5
    
    model = DNN(inputs_dim, hidden_units, activation=activation, 
        l2_reg=l2_reg, dropout_rate=dropout_rate, use_bn=use_bn, dice_dim=2)

    mod = PyTorchModule("deepctr_dnn", model)

    input_shape = (1, inputs_dim)

    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


# TODO: When microbatch > 1 is supported in backend convert this to a verify_module test
@pytest.mark.parametrize("use_bias", [True, False], ids=['bias', 'no_bias'])
@pytest.mark.parametrize("task", ['binary', 'regression'])
def test_prediction_layer(test_kind, test_device, task, use_bias):
    if test_kind.is_training() and use_bias:
        pytest.skip("Cannot calculate gradient of input when modified in place")

    model = PredictionLayer(task, use_bias)
    mod = PyTorchModule("deepctr_prediction_layer", model)

    input_shape = (1, 64, 64)
    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


in_channels = [16, 32]
out_channels = [64, 128]
kernel_sizes = [2, 3]
strides = [1, 2, 3]
biases = [True, False]

# NOTE: Non constant pad size is not supported, all tests will fail
# TODO: When microbatch > 1 is supported in backend convert this to a verify_module test
# @pytest.mark.parametrize("bias", biases)#, ['bias', 'no_bias'])
# # @pytest.mark.parametrize("dilation", dilations)#, [f"dilation{str(s)}" for s in dilations])
# @pytest.mark.parametrize("stride", strides)#, [f"stride{str(s)}" for s in strides])
# @pytest.mark.parametrize("kernel_size", kernel_sizes)#, [f"kernel{str(s)}" for s in kernel_sizes])
# @pytest.mark.parametrize("out_channels", out_channels)#, [f"out{str(s)}" for s in out_channels])
# @pytest.mark.parametrize("in_channels", in_channels)#, [f"in{str(s)}" for s in in_channels])
# def test_conv2d_same(training, in_channels, out_channels, kernel_size, stride, bias):
    
#     pytest.skip("Non constant pad size not supported")

#     recompute = True
#     compile_depth = CompileDepth.FULL

#     model = Conv2dSame(in_channels, out_channels, kernel_size, stride, bias=bias)

#     mod = PyTorchModule("deepctr_conv2d_same", model)

#     sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
#     tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
#     tt0.place_module(mod)

#     inp = torch.randn((64, in_channels, 64, 64))

#     ret = pybuda_compile(
#         tt0,
#         "deepctr_conv2d_same",
#         inp,
#         compiler_cfg=CompilerConfig(
#             enable_training=training,
#             enable_recompute=recompute,
#             compile_depth=compile_depth,
#         ),
#         verify_cfg=VerifyConfig(
#             intermediates=True,
#         ),
#     )
#     evaluate_framework_vs_pybuda(model, ret, inp)
