# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
import torch.nn as nn
from PIL import Image

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model
import pybuda

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.f = -1
        self.i = -1

    def forward(self, x):
        return x


def test_yolov5_320x320(test_kind, test_device):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()
    
    # This one works
    
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    # pytest.skip("Last detect module has 5D shapes, skip for now")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tm_cpu_fallback = True
    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5s", pretrained=True)

    module = PyTorchModule("pt_yolov5s", model)

    input_shape = (1, 3, 320, 320)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework = True,
        ),
    )
    
    
def test_yolov5_480x480(test_kind, test_device):
    
    pytest.skip("Needs padding")
    
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    # pytest.skip("Last detect module has 5D shapes, skip for now")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    # Add required env vars as per: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/model-demos/-/issues/46
    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16}"
    os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_tm_cpu_fallback = True

    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5s", pretrained=True)

    module = PyTorchModule("pt_yolov5s", model)

    input_shape = (1, 3, 480, 480)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
            run_golden=True
        ),
    )



def test_yolov5m_640x640(test_kind, test_device):
    
    pytest.skip("Needs padding")
    
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    # pytest.skip("Last detect module has 5D shapes, skip for now")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    # Add required env vars as per: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/model-demos/-/issues/46
    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16, 3:4}"
    os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"
    os.environ["PYBUDA_CONCAT_SLICE_Y"] = "8"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_tm_cpu_fallback = True
    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5m", pretrained=True)

    module = PyTorchModule("pt_yolov5m_640_640", model)

    input_shape = (1, 3, 640, 640)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    
    
def test_yolov5_1280x1280(test_kind, test_device):
    
    pytest.skip("Needs padding")
    
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    # pytest.skip("Last detect module has 5D shapes, skip for now")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    # Add required env vars as per: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/model-demos/-/issues/46
    import os
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{13:16}"
    os.environ["PYBUDA_INSERT_SLICE_FOR_CONCAT"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_tm_cpu_fallback = True

    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5s", pretrained=True)

    module = PyTorchModule("pt_yolov5s", model)

    input_shape = (1, 3, 1280, 1280)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
            run_golden=True
        ),
    )
    
