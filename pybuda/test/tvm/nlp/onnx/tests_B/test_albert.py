# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from distutils.command.config import config
import pybuda
import pytest

import torch
from transformers import AlbertConfig, AlbertModel

from pybuda import (
    OnnxModule,
    TTDevice,
    CPUDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import onnx
import os

def test_albert_v2(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 768, 768)

    model = AlbertModel.from_pretrained("albert-base-v2", torchscript=True)

    submodel = model.encoder.albert_layer_groups[0].albert_layers[0]
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/albert_v2_layer.onnx"

    torch.onnx.export(submodel,               # model being run
                        torch.randn(input_shape),                         # model input (or a tuple for multiple inputs)
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
        "albert_v2_layer_onnx",
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
            waive_gradient_errors={"attention.key.bias"},
        )
    )
    os.remove(save_path)
