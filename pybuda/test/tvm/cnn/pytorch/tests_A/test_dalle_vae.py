# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest
import os

import torch
import torch.nn as nn
from test.tvm.cnn.pytorch.dall_e_vae import Encoder, Decoder


from pybuda import (
    PyTorchModule,
    VerifyConfig,
)

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import pybuda


def test_tvm_dalle_Encoder(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    input_shape = (1, 3, 224, 224)

    model = Encoder()
    mod = PyTorchModule("DALLE_vae_encoder", model.blocks[:3])  # Reduce compile time

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_dalle_Decoder(test_kind, test_device):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON2"] = "1"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    input_shape = (1, 8192, 32, 32)

    model = Decoder()
    mod = PyTorchModule("DALLE_vae_encoder", model.blocks[:3])  # Reduce compile time

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
