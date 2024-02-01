# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# GPT Neo basic bring-up tests of tracing functionality
#
from pybuda._C.backend_api import BackendDevice
import pytest

import torch
from transformers import GPTNeoModel, GPTNeoConfig
import os

from pybuda import (
    PyTorchModule,
    CompileDepth,
    VerifyConfig,
    BackendType,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_gptneo_block(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    torch.manual_seed(52)
    input_shape = (1, 64, 2560)
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B", torchscript=True)
    config.num_layers = 1  # For faster model loading
    model = GPTNeoModel(config)
    submodel = model.h[0]
    mod = PyTorchModule("gptneo_block", submodel)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        uniform_inputs=True,
    )

def test_gptneo_full(test_kind, test_device):

    # Pipegen error on silicon if enabled
    os.environ["PYBUDA_DISABLE_STABLE_SOFTMAX"] = "1"
    os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "100000"
    
    if test_kind == TestKind.TRAINING:
        pytest.skip()
    
    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER 

    #Fusing disabled due to tenstorrent/pybuda#789
    if test_kind == TestKind.INFERENCE and test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.enable_auto_fusing=False

    torch.manual_seed(52)
    input_shape = (1, 256)
    inputs = [torch.randint(0, input_shape[-1], input_shape)]
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B", torchscript=True)
    config.num_layers = 1  # For faster model loading
    model = GPTNeoModel(config)
    mod = PyTorchModule("gptneo_full", model)

    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        mod,
        (input_shape,),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=pcc,
        ),
        uniform_inputs=True,
    )

    os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "0"
