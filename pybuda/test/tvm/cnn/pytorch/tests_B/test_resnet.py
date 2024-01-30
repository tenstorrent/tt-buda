# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch

import pybuda
from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.utils import download_model


def get_relaxed_atol_pcc(test_kind, test_device, microbatch_size=1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.95
    if test_device.is_silicon():
        training_pcc = 0.8
        training_atol = 0.55
    inference_atol = 0.1
    inference_pcc = 0.95
    relative_atol = training_atol if test_kind.is_training() else inference_atol
    if test_device.is_silicon() and test_kind.is_training():
        relative_atol *= 3.5
    pcc = training_pcc if test_kind.is_training() else inference_pcc

    return relative_atol, pcc


def test_resnet_pytorch(test_kind, test_device):
    # Always run with recompute in post-commit CI. Nightly tests both
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
        # compiler_cfg.compile_depth = CompileDepth.FULL
    compiler_cfg.balancer_policy = "CNN"
    # compiler_cfg.place_on_new_epoch("max_pool2d_14.dc.reshape.0_operand_commute_clone411.dc.sparse_matmul.4.lc2")

    # Issue below is still valid, though it doesn't trigger when fracturing is turned on
    # tenstorrent/pybuda#310
    #pybuda.config.override_t_stream_shape(
    #    "conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (28, 1)
    #)

    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    module = PyTorchModule("pt_resnet", model)

    input_shape = (1, 3, 224, 224)
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            relative_atol=relative_atol,
            pcc=pcc,
        ),
    )


def test_resnet_pytorch_instance_norm(test_kind, test_device):
    pytest.skip()  # WIP

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    # TODO: Remove
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # compiler_cfg.compile_depth = CompileDepth.BALANCER_PASS
        compiler_cfg.compile_depth = CompileDepth.FULL
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.place_on_new_epoch("conv2d_0.dc.reshape.15.dc.sparse_matmul.1.lc2")

    from torchvision.models import resnet18
    model = resnet18(norm_layer=torch.nn.InstanceNorm2d)
    module = PyTorchModule("pt_resnet_instance_norm", model)

    input_shape = (1, 3, 224, 224)
    out = model(torch.rand(input_shape))

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            relative_atol=relative_atol,
            pcc=pcc,
        ),
    )
