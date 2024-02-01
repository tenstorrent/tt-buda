# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.config import CompileDepth
import pytest

import torch
import torch.nn as nn
# from transformers.models.squeezebert import SqueezeBertEncoder
from transformers import SqueezeBertModel, SqueezeBertConfig

import math
import itertools
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    OnnxModule,
    optimizers,
    pybuda_compile,
    tvm_to_python,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
import os
import onnx
import onnxruntime as ort

def test_tvm_SqueezeBertEncoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER 

    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    input_shape = (1, 32, 768)

    config = SqueezeBertConfig()
    config.num_hidden_layers = 1
    model = SqueezeBertModel(config)

    attention_mask = torch.ones(input_shape[0:2])
    extended_attn_mask = model.get_extended_attention_mask(attention_mask, input_shape[0:2], "cpu")

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/SqueezeBertEncoder_onnx.onnx"

    torch.onnx.export(model.encoder,               # model being run
                        tuple([torch.randn(input_shape), extended_attn_mask]), # model input (or a tuple for multiple inputs),
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
        "SqueezeBertEncoder_onnx",
        onnx_model,
        save_path,
    )

    verify_module(
        mod,
        (input_shape,extended_attn_mask.shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    os.remove(save_path)




def test_tvm_SqueezeBertPooler(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER 

    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    config = SqueezeBertConfig()

    model = SqueezeBertModel(config)

    input_shape = (1, 8, 768)

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/SqueezeBertPooler_onnx.onnx"

    traced_model = torch.jit.trace(model.pooler, tuple([torch.randn(input_shape),]), strict=False)

    torch.onnx.export(traced_model,               # model being run
                        tuple([torch.randn(input_shape),]), # model input (or a tuple for multiple inputs),
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
        "SqueezeBertPooler_onnx",
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
