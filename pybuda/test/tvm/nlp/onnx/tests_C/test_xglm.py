# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda._C.backend_api import BackendDevice
from pybuda.config import CompileDepth
from pybuda.verify.config import TestKind
import pytest

import torch
from transformers import XGLMModel, XGLMConfig
from pybuda import (
    PyTorchModule,
    OnnxModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    tvm_to_python,
)
from pybuda.config import CompileDepth, _get_global_compiler_config

import os
import onnx
from pybuda.verify import verify_module

def test_tvm_xglm(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    config = XGLMConfig()
    input_shape = (1, 32, 1024)
    config.num_layers = 1

    model = XGLMModel(config)

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/xglm_onnx.onnx"

    torch.onnx.export(model.layers[0],               # model being run
                        torch.randn( input_shape), # model input (or a tuple for multiple inputs),
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "xglm_onnx",
        onnx_model,
        save_path,
    )

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    os.remove(save_path)


def test_tvm_xglm_fallback(test_kind, test_device):
    # Training fails due to unsupported op
    if test_kind.is_training():
        pytest.skip()

    config = XGLMConfig()
    input_shape = (1, 32,)
    config.num_layers = 1

    model = XGLMModel(config)

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/xglm_onnx.onnx"

    torch.onnx.export(model,               # model being run
                        torch.randint(2000, input_shape), # model input (or a tuple for multiple inputs),
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "xglm_onnx",
        onnx_model,
        save_path,
    )
    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.retain_tvm_python_files = True

    verify_module(
        mod,
        (input_shape,),
        input_params=[{"data_format" :  torch.int64},],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    os.remove(save_path)
