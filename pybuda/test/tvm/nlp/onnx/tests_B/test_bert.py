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
from test.backend.models.test_bert import get_relaxed_atol_pcc

import urllib
import os




from transformers import BertModel, BertConfig, BertForPreTraining


from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

def test_bert_encoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 128, 128)
    model = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)

    submodel = model.encoder
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/bert_tiny.onnx"

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
        "bert_tiny_onnx",
        onnx_model,
        save_path,
    )

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            relative_atol=relative_atol, 
            pcc=pcc,
            waive_gradient_errors={"attention.self.key.bias"},
            enable_input_gradient_checking=False,
        ),
    )
    os.remove(save_path)
