# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# UniSpeech basic bring-up tests of tracing functionality
#
import pytest

import torch
from transformers import UniSpeechModel

from pybuda import (
    OnnxModule,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import os
import onnx


def test_unispeech(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    else:
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER  # Unsupported HW ops

    framework_model = UniSpeechModel.from_pretrained(
        "microsoft/unispeech-sat-base", torchscript=True
    )
    input_shape = (1, 512)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/unispeech_onnx.onnx"

    torch.onnx.export(framework_model,               # model being run
                        torch.randn(input_shape), # model input (or a tuple for multiple inputs),
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
        "unispeech_onnx",
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
        ),
    )
    os.remove(save_path)
