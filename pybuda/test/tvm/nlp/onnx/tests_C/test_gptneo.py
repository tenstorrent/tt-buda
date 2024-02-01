# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from transformers import GPTNeoModel, GPTNeoConfig

from pybuda import (
    PyTorchModule,
    CompileDepth,
    VerifyConfig,
    OnnxModule,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import onnx
import os


def test_gptneo_onnx(test_kind, test_device):
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
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/gptneo_onnx.onnx"
    traced_model = torch.jit.trace(submodel, tuple([torch.randn(input_shape),]), strict=False)
    torch.onnx.export(traced_model,               # model being run
                        torch.randn(input_shape), # model input (or a tuple for multiple inputs),
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
        "gptneo_onnx",
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

