# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.tvm.cnn.pytorch.gscnn import get_model
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


def test_gscnn_pytorch(test_kind, test_device):
    pytest.skip() # Takes too long to compile/run
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    
    model = download_model(get_model, 
        network="gscnn.gscnn.GSCNN",
        num_classes=30,
        criterion=None, # Only needed for training
        trunk="resnet18",
    )

    module = PyTorchModule("gscnn_torch", model)

    input_shape = (1, 3, 1024, 2048)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

from test.tvm.cnn.pytorch.gscnn.wider_resnet import wider_resnet38_a2

def test_wider_resnet_torch(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    model = wider_resnet38_a2(classes=1000, dilation=True)
    submodel = torch.nn.Sequential(
        model.mod1,
        model.pool2,
    )

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH # Needs neg maxpool support tenstorrent/pybuda#188

    module = PyTorchModule("wider_resnet_torch", submodel)

    input_shape = (1, 3, 1024, 2048)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


import test.tvm.cnn.pytorch.gscnn.gated_spatial_conv as gsc

def test_gated_spatial_conv_torch(test_kind, test_device):
    pytest.skip() #TODO: Debug why this runs out of memory
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    model = gsc.GatedSpatialConv2d(32, 32)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS # tenstorrent/pybuda#185

    module = PyTorchModule("gated_spatial_conv_torch", model)

    input_shape0 = (1, 32, 1024, 2048)
    input_shape1 = (1, 1, 1024, 2048)
    verify_module(
        module,
        (input_shape0, input_shape1),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


from test.tvm.cnn.pytorch.gscnn.gscnn import _AtrousSpatialPyramidPoolingModule
# Need to support non-square pooling + convolution (kernel size and stride)
def test_spatial_pyramid_pooling_torch(test_kind, test_device):
    pytest.skip()

    model = _AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    module = PyTorchModule("spatial_pyramid_pooling_torch", model)

    input_shape0 = (1, 4096, 128, 256)
    input_shape1 = (1, 1, 1024, 2048)
    verify_module(
        module,
        (input_shape0, input_shape1),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
