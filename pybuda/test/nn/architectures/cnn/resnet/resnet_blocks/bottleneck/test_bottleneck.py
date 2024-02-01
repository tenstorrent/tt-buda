# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
#

import os
import pytest
import torch
import numpy as np

import pybuda
import pybuda.op
from pybuda import Tensor
from pybuda import TTDevice, BackendType, pybuda_compile, VerifyConfig, CompilerConfig

from . import TestBottleneckBlock

kernel = list(range(1, 7))
channels = [16, 32]
inbetween_channels = [32]
original_shapes = [(32, 32), (36, 37), (62, 55)]
stride = list(range(1, 5))
dilation = list(range(1, 2))

@pytest.mark.xfail
@pytest.mark.parametrize("in_out_channels", channels, ids=[f"channel{ch}" for ch in channels])
@pytest.mark.parametrize("inbetween_channels", inbetween_channels, ids=[f"inbtw{ich}" for ich in inbetween_channels])
@pytest.mark.parametrize("original_shape", original_shapes, ids=[f"orig_shape{'x'.join([str(item) for item in osh])}" for osh in original_shapes])
@pytest.mark.parametrize("stride", stride, ids=[f"stride{st}" for st in stride])
@pytest.mark.parametrize("dilation", dilation, ids=[f"dilation{d}" for d in dilation])
@pytest.mark.parametrize("depthwise", [False]) #True, False], ids=["Depthwise", "NoDepthwise"])
@pytest.mark.parametrize("bias", [True, False], ids=["Bias", "NoBias"])
@pytest.mark.parametrize("padding_mode", ["zeros"])
@pytest.mark.parametrize("recompute", [True, False], ids=["Recompute", "NoRecompute"])
@pytest.mark.parametrize("mode", ["Training", "Inference"])
def test_bottleneck_block(
    mode,
    recompute,
    in_out_channels,
    inbetween_channels,
    original_shape,
    stride,
    dilation,
    depthwise,
    bias,
    padding_mode
):

    training = (mode == "Training")

    if training or recompute:
        pytest.skip("Backward Pass - Not Implmented.")

    model = TestBottleneckBlock(
        in_channels=in_out_channels,
        out_channels=in_out_channels,
        inbetween_channels=inbetween_channels,
        stride=stride,
        dilation=dilation,
        depthwise=depthwise,
        bias=bias
    )

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)

    activations = Tensor.create_from_torch(
        torch.rand(
            1, 
            in_out_channels,
            original_shape[0], 
            original_shape[1], 
            requires_grad=True
        )
    )
    _, _, _, outputs, _ = pybuda_compile(
        tt0,
        "conv2d",
        activations,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(),
    )