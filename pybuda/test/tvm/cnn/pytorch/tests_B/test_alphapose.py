# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
import os

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

from test.tvm.cnn.pytorch.alphapose.utils.config import update_config
from test.tvm.cnn.pytorch.alphapose.models import builder

def test_alphapose(test_kind, test_device):
    pytest.skip("Has non-singleton 6D shapes, skip for now")
    if test_kind == TestKind.TRAINING:
        pytest.skip()
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../alphapose"
    cfg = update_config(dir_path + "/256x192_res50_lr1e-3_1x.yaml")

    model = builder.build_sppe(cfg["MODEL"], preset_cfg=cfg["DATA_PRESET"])

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    module = PyTorchModule("alphapose", model)

    input_shape = (1, 3, 256, 192)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )
