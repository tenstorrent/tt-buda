# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest

import torch
import torch.nn as nn
from transformers.models.xlm import XLMConfig, XLMModel, XLMPreTrainedModel

import math
import itertools
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    tvm_to_python,
    OnnxModule,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import os
import onnx

def test_tvm_xlm_FFN(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    config = XLMConfig()

    model = XLMModel(config)

    input_shape = (1, 16, 2048)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/xlm_FFN_onnx.onnx"

    torch.onnx.export(model.ffns[0],               # model being run
                        torch.rand(input_shape), # model input (or a tuple for multiple inputs),
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "xlm_FFN_onnx",
        onnx_model,
        save_path,
    )

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        )
    )
    os.remove(save_path)
