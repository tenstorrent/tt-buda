# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.verify.backend import verify_module
from test.tvm.recommendation.pytorch.deepctr_torch.layers.interaction import CIN, FM, AFMLayer, BiInteractionPooling, BilinearInteraction, ConvLayer, CrossNet, CrossNetMix, InnerProductLayer, InteractingLayer, LogTransformLayer, OutterProductLayer
from tvm.contrib.pybuda_compile import compile_tf_graphdef_for_buda
from deepctr_torch.layers.interaction import *
import torch
from torch import nn
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
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.tvm.utils import evaluate_framework_vs_pybuda
import pytest


import torch


# TODO: Figure out why this test can't have batch_size > 1 but others can
def test_FM(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.FULL

    batch_size = 1
    field_size = 32
    embedding_size = 64
    input_shape = (batch_size, field_size, embedding_size)

    model = FM()
    mod = PyTorchModule("deepctr_fm", model)

    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


# TODO: This test remains as a pybuda_compile test rather than verify_module test because batch_size > 1 fails for verify_module
def test_bilinear_interaction(training):

    pytest.skip("Test hangs on GENERATE_INITIAL_GRAPH")

    recompute = True

    compile_depth = CompileDepth.FULL

    batch_size = 20
    filed_size = 32
    embedding_size = 64
    input_shape = (batch_size, filed_size, embedding_size)

    class BilinearInteractionWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = BilinearInteraction(filed_size, embedding_size)

        def forward(self, x):
            return self.model(x)

    model = BilinearInteractionWrapper()
    mod = PyTorchModule("deepctr_bilinear_interaction", model)

    inp = torch.rand(input_shape)
    
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "deepctr_bilinear_interaction",
        inp,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, ret, inp)



# TODO: Figure out why this test can't have batch_size > 1 but others can
def test_cin(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    if test_kind.is_training():
        _get_global_compiler_config().compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    batch_size = 1
    field_size = 32
    embedding_size = 64
    input_shape = (batch_size, field_size, embedding_size)

    model = CIN(field_size)
    mod = PyTorchModule("deepctr_cin", model)

    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

# TODO: Figure out why this test can't have batch_size > 1 but others can
def test_afm_layer(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    if test_kind.is_training():
        _get_global_compiler_config().compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    batch_size = 1
    in_features = 1
    embedding_size = 64
    input_shape = (batch_size, in_features, embedding_size)

    model = AFMLayer(in_features)
    mod = PyTorchModule("deepctr_afm_layer", model)
    inputs = [[torch.randn(input_shape), torch.randn(input_shape), torch.randn(input_shape), torch.randn(input_shape)]]
    verify_module(
        mod,
        (),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_interacting_layer(training):
    pytest.skip("operator stack is not supported")

    recompute = True

    compile_depth = CompileDepth.FULL
        
    batch_size = 20
    field_size = 32
    embedding_size = 64
    input_shape = (batch_size, field_size, embedding_size)

    model = InteractingLayer(embedding_size)
    mod = PyTorchModule("deepctr_interacting_layer", model)

    inp = torch.rand(input_shape)
    
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "deepctr_interacting_layer",
        inp,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, ret, inp)


# TODO: Figure out why this test can't have batch_size > 1 but others can
def test_crossnet(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
        
    batch_size = 1
    units = 32

    input_shape = (batch_size, units)

    model = CrossNet(units)
    mod = PyTorchModule("deepctr_crossnet", model)

    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_crossnet_mix(training):
    pytest.skip("Operator stack not suported")
    
    recompute = True

    compile_depth = CompileDepth.FULL
        
    batch_size = 20
    units = 32

    input_shape = (batch_size, units)

    model = CrossNetMix(units)
    mod = PyTorchModule("deepctr_crossnet_mix", model)

    inp = torch.rand(input_shape)
    
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "deepctr_crossnet_mix",
        inp,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, ret, inp)


def test_inner_product_layer(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip("concatenate backward not implemented in op/eval/pybuda/eltwise_nary.py")

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    
    batch_size = 20
    embedding_size = 32

    input_shape = (batch_size, 1, embedding_size)

    model = InnerProductLayer()
    mod = PyTorchModule("deepctr_inner_product_layer", model)
    inputs = [[torch.randn(input_shape), torch.randn(input_shape)]]
    verify_module(
        mod,
        (),
        inputs = [inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


# TODO: Figure out why this test can't have batch_size > 1 but others can
def test_outer_product_layer(test_kind, test_device):


    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.PRE_LOWERING_PASS # Doesnt support broadcasting on w

    batch_size = 1
    embedding_size = 32
    field_size = 16

    input_shape = (batch_size, 1, embedding_size)

    class OuterProductLayerWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = OutterProductLayer(field_size, embedding_size)

        def forward(self, *x):
            return self.model(x)

    model = OuterProductLayerWrapper()
    mod = PyTorchModule("deepctr_outer_product_layer", model)

    verify_module(
        mod,
        [input_shape, input_shape],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_conv_layer(training):
    pytest.skip("Uses Conv2DSame submodules, which are not supported currently. See test_core_modules.py for more info")

    recompute = True

    compile_depth = CompileDepth.FULL

    batch_size = 20
    embedding_size = 32
    field_size = 16
    conv_kernel_width = [2]
    conv_filters = [16]

    input_shape = (batch_size, 1, field_size, embedding_size)

    class ConvWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ConvLayer(field_size, conv_kernel_width, conv_filters)

        def forward(self, x):
            return self.model(x)

    model = ConvWrapper()
    mod = PyTorchModule("deepctr_conv_layer", model)

    inp = torch.rand(input_shape)
    
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    ret = pybuda_compile(
        tt0,
        "deepctr_conv_layer",
        inp,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, ret, inp)


def test_bi_interaction_pooling(test_kind, test_device):

    
    batch_size = 20
    field_size = 32
    embedding_size = 64
    input_shape = (batch_size, field_size, embedding_size)

    model = BiInteractionPooling()
    mod = PyTorchModule("deepctr_bi_interaction_pooling", model)
    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


@pytest.mark.skip(reason="Clip not supported in backend")
def test_log_transform_layer(test_kind, test_device):
    
    recompute = True

    compile_depth = CompileDepth.FULL
        
    batch_size = 30
    field_size = 32
    embedding_size = 64
    ltl_hidden_size = 128

    input_shape = (batch_size, field_size, embedding_size)

    model = LogTransformLayer(field_size, embedding_size, ltl_hidden_size)
    mod = PyTorchModule("deepctr_log_transform_layer", model)

    verify_module(
        mod,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
