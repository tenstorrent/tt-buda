# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# DETR basic bring-up tests of tracing functionality
#
import os
import pytest

import torch
from transformers import DetrConfig, DetrModel

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.utils import download_model


def test_detr_50_full(test_kind, test_device):
    # tenstorrent/pybuda#392
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

    if test_kind.is_training():
        # Training is currently unsupported
        pytest.skip()

    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Wrapper(torch.nn.Module):
        def __init__(self, model, act_shape):
            super().__init__()
            self.model = model

            batch_size, num_channels, height, width = act_shape
            self.pixel_mask = torch.ones((batch_size, height, width))

        def forward(self, hidden_states):
            return self.model(hidden_states, self.pixel_mask)

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.balancer_policy = "CNN"

    # Inputs
    input_shape = (1, 3, 256, 256)

    # Configure PyTorch module
    framework_module = download_model(
        DetrModel.from_pretrained,
        "facebook/detr-resnet-50", torchscript=True
    )
    framework_module = Wrapper(framework_module, input_shape)
    pybuda_module = PyTorchModule(
        "pt_detr50",
        framework_module,
    )

    # Run module
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.99,
        ),
    )


def test_detr_18_full(test_kind, test_device):
    # Running full DeTr model with Resnet50 as backbone. No need
    # to run this variation separately in CI at the moment.
    pytest.skip()
    
    # Doesn't use pre-trained weights, but randomly initialized
    # ones. Reason is that resnet18 is not available by default
    # as backbone model, therefore, there aren't any pre-trained
    # weights for it.

    if test_kind.is_training():
        # Training is currently unsupported due to the CNN backbone
        pytest.skip()

    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Wrapper(torch.nn.Module):
        def __init__(self, model, act_shape):
            super().__init__()
            self.model = model

            batch_size, num_channels, height, width = act_shape
            self.pixel_mask = torch.ones((batch_size, height, width))

        def forward(self, hidden_states):
            return self.model(hidden_states, self.pixel_mask)

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL 
    # compiler_cfg.retain_tvm_python_files = True
    # compiler_cfg.enable_tvm_unsupported_ops = True
    compiler_cfg.enable_tvm_constant_prop = True
    # compiler_cfg.cpu_fallback_ops.add("zeros")
    # verify_cfg.verify_pybuda_codegen_vs_framework = False  # PCC is over 0.992

    # Inputs
    input_shape = (1, 3, 256, 256)

    # Configure PyTorch module
    framework_config = DetrConfig()
    framework_config.backbone = "resnet18"
    framework_module = DetrModel(framework_config)

    framework_module = Wrapper(framework_module, input_shape)
    pybuda_module = PyTorchModule(
        "pt_detr18",
        framework_module,
    )

    # Run module
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
        ),
    )








