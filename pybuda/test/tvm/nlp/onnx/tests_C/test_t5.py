# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import onnx
import onnxruntime as ort
import pytest
import torch
from pybuda import (
    OnnxModule,
    TTDevice,
    BackendType,
    PyTorchModule,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    TFGraphDefModule,
)
from pybuda.config import CompileDepth
from loguru import logger
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from transformers import T5Config, T5Model, T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel

import urllib
import os

def test_t5_small_decoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 1, 128, 512)

    compiler_cfg = _get_global_compiler_config()

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    pretrained_name = "t5-small"
    config = T5Config.from_pretrained(pretrained_name, torchscript=True)
    model = T5Model(config)
    pretrained_model = T5Model.from_pretrained(pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/t5_small_decoder.onnx"

    torch.onnx.export(model.decoder.block[0],               # model being run
                        torch.randn(input_shape),# model input (or a tuple for multiple inputs),
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=14,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "t5_small_decoder_onnx",
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

# Unsupported HW op transpose XZ
@pytest.mark.skip(reason="Found unsupported HW ops, stopping compilation early: transpose_632 transpose(dim0: 1, dim1: 3, z_dim_slice: 12)")
def test_tvm_t5_onnx(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    save_path = os.path.dirname(os.path.realpath(__file__)) + "t5-encoder-12.onnx"

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/raw/main/text/machine_comprehension/t5/model/t5-encoder-12.onnx",
            save_path,
        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "t5_onnx",
        onnx_model,
        save_path,
    )

    compiler_cfg = _get_global_compiler_config()
    # compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER 
    # compiler_cfg.tvm_graph_load_path = "onnx_t5_fallback"
    compiler_cfg.retain_tvm_python_files = True
    # compiler_cfg.enable_tvm_unsupported_ops = True
    input_shape = (1, 8)
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


def test_t5_small_fallback(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class T5Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, decoder_input_ids):
            return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    # Define compiler configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True 

    # Load pre-trained model and config
    pretrained_name = "t5-small"
    config = T5Config.from_pretrained(pretrained_name)
    config.use_cache = False
    model = T5Model(config)
    pretrained_model = T5Model.from_pretrained(pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())
    model = T5Wrapper(model)

    # Export ONNX model
    input_shape = (1, 128)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/t5_small.onnx"
    torch.onnx.export(
        model,  # model being run
        tuple(
            [torch.randint(2000, input_shape), torch.randint(2000, input_shape)]
        ),  # model input (or a tuple for multiple inputs),
        save_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )

    # Load ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "t5_small_onnx",
        onnx_model,
        save_path,
    )

    # Compile and verify
    verify_module(
        mod,
        (input_shape, input_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        input_params=[
            {"requires_grad": False, "data_format": torch.int64},
            {"requires_grad": False, "data_format": torch.int64},
        ],
    )

    # Cleanup
    os.remove(save_path)
