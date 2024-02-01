# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
    CompileDepth,
)
from pybuda.verify.backend import verify_module
from pybuda.config import _get_global_compiler_config
from pybuda.verify.config import TestKind

import timm

def test_regnety_002_pytorch(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    model = model = timm.create_model('regnety_002', pretrained=True)
    module = PyTorchModule("regnety_002", model)
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    
    input_shape = (1, 3, 224, 224)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
