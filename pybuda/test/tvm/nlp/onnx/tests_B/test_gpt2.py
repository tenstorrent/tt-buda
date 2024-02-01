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
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    TFGraphDefModule,
)
from pybuda.config import CompileDepth

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

import urllib
import os
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


@pytest.mark.skip(reason="Pretrained Onnx gpt2 has 5D shape.")
def test_tvm_gpt2_onnx(test_kind, test_device):

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/gpt2.onnx"

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
            save_path,
        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "gpt2_onnx",
        onnx_model,
        save_path,
    )

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"} 
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_graph_load_path = "onnx_gpt2_fallback"

    input_shape = (1, 1, 8)
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


def test_tvm_gpt2_block(test_kind, test_device):
    pytest.skip("Segmentation fault, skipping for now.")
    
    # Training without TVM constant prop will result in the following error in placer
    #   RuntimeError: trying to place bw_in0_gpt2_block.attn_c_attn_weight_combine_add_0_transpose_nop
    #   and exceeded max-placement-attempts: grid_shape: (2, 8, original context.start=(.row=5, .col = 5)
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    # Load pretrained model
    model = GPT2Model.from_pretrained("gpt2")
    submodule = model.h[0]

    # Input shape
    input_shape = (1, 64, 768)
    act_input = torch.randn(input_shape)
    out_expected = submodule(act_input)

    # Export to ONNX
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/got2_block.onnx"
    torch.onnx.export(
        submodule,
        (act_input,),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    onnx_model = OnnxModule(
        "gpt2_block_onnx",
        onnx_model,
        save_path,
    )

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask = {"attn.c_attn.weight", "attn.c_attn.bias"}

    # Atol & PCC
    relative_atol = 0.4 if test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.9 if test_device.devtype == BackendType.Silicon else 0.99

    verify_module(
        onnx_model,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            relative_atol=relative_atol,
            waive_gradient_errors={"c_attn.bias"},
            pcc=pcc,
        ),
    )

    # Cleanup
    os.remove(save_path)


def test_tvm_gpt2_layer_onnx_fallback(test_kind, test_device):
    pytest.skip("Mismatched attribute type during ONNX export.")

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn.c_attn.weight", "attn.c_attn.bias"}  

    input_shape = (1, 64)
   
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 1
    config.use_cache = False
    model = GPT2Model(config)

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/gpt2.onnx"

    torch.onnx.export(model,               # model being run
                        torch.randint(2000, input_shape),                         # model input (or a tuple for multiple inputs)
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
        "gpt2_onnx",
        onnx_model,
        save_path,
    )

    input_shape = (1, 64)
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
