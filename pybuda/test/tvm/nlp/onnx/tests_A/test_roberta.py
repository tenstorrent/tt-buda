# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import onnx
import onnxruntime as ort
from pybuda._C.backend_api import BackendDevice
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
from test.backend.models.test_bert import get_relaxed_atol_pcc
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

import urllib
from loguru import logger
import os
from transformers import RobertaModel, RobertaConfig


@pytest.mark.skip(reason="roberta casts int64 to bool")
def test_tvm_roberta_layer(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    config = RobertaConfig()
    config.num_hidden_layers = 1
    config.use_cache = False

    model = RobertaModel(config, add_pooling_layer=False)

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/roberta_layer.onnx"

    input_shape = (1, 128)
    torch.onnx.export(model,               # model being run
                        torch.randint(config.vocab_size, input_shape),                         # model input (or a tuple for multiple inputs)
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "roberta_layer_onnx",
        onnx_model,
        save_path,
    )


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

# This test fails after lowering to buda due to output mismatch
# def test_tvm_roberta_onnx(test_kind, test_device):
#     if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
#         pytest.skip()

#     if test_kind.is_training():
#         pytest.skip()
#         test_device.devtype = BackendType.NoBackend
#     save_path = os.path.dirname(os.path.realpath(__file__)) + "/roberta-base-11.onnx"

#     if not os.path.exists(save_path):
#         urllib.request.urlretrieve(
#             "https://github.com/onnx/models/raw/main/text/machine_comprehension/roberta/model/roberta-base-11.onnx",
#             save_path,
#         )

#     onnx_model = onnx.load(save_path)
#     onnx.checker.check_model(onnx_model)
#     mod = OnnxModule(
#         "roberta_onnx",
#         onnx_model,
#         save_path,
#     )

#     compiler_cfg = _get_global_compiler_config()
#     # compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
#     compiler_cfg.enable_tvm_cpu_fallback = True
#     compiler_cfg.tvm_graph_store_path = "onnx_roberta_fallback"
#     # compiler_cfg.enable_tvm_unsupported_ops = True
#     input_shape = (1, 8)
#     verify_module(
#         mod,
#         (input_shape,),
#         input_params=[{"data_format" :  torch.int64},],
#         verify_cfg=VerifyConfig(
#             arch=test_device.arch,
#             devtype=test_device.devtype,
#             test_kind=test_kind,
#         )
#     )
#     os.remove(save_path)


def test_tvm_roberta(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    input_shape = (1, 256, 256)
    roberta_model = RobertaModel.from_pretrained("arampacha/roberta-tiny", torchscript=True)
    model = roberta_model.encoder

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/roberta_encoder_onnx.onnx"

    torch.onnx.export(model,               # model being run
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
        "roberta_encoder_onnx",
        onnx_model,
        save_path,
    )

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, "tiny", 1)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"attention.self.key.bias", "attention.self.query.bias", "453", "454"},
            relative_atol=relative_atol,
            pcc=pcc,
        )
    )
    os.remove(save_path)
