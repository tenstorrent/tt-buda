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
import torch.nn as nn


def test_gelu_onnx(test_device):
    class GELUActivation(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
        the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, input):
            return torch.nn.functional.gelu(input)

    model = GELUActivation()
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/onnx_gelu.onnx"
    input_shape = (1, 64, 3072)
    torch.onnx.export(model,               # model being run
                        torch.randn(input_shape),# model input (or a tuple for multiple inputs),
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
        "onnx_gelu",
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


def test_new_gelu_onnx(test_device):
    import math
    class NewGELUActivation(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
        the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, input):
            return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    model = NewGELUActivation()

    input_shape = (1, 64, 3072)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/onnx_gelu_approx.onnx"

    torch.onnx.export(model,               # model being run
                        torch.randn(input_shape),# model input (or a tuple for multiple inputs),
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
        "onnx_gelu_approx",
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


input_shapes = [(1, 512, 10, 10)]
pool_out_sizes = [(1, 1)]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "pool_out_size", pool_out_sizes, ids=[f"pouts({str(p)})" for p in pool_out_sizes]
)
def test_tvm_avg_pool_onnx(test_kind, test_device, input_shape, pool_out_size):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class AdaptiveAvgPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(pool_out_size)

        def forward(self, a):
            b = self.avgpool(a)
            c = torch.flatten(b, 1)

            return c

    model = AdaptiveAvgPool()
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/avg_pool_onnx.onnx"

    torch.onnx.export(model,               # model being run
                        torch.randn(input_shape),# model input (or a tuple for multiple inputs),
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
        "avg_pool_onnx",
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

def test_tvm_emb_linear_onnx_fallback(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class EmbModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50000, 32)
            self.linear = nn.Linear(32, 32)

        def forward(self, input):
            embs = self.emb(input)
            lin = self.linear(embs)
            return lin

    input_shape = (1, 32)
    model = EmbModel()
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/emb_linear.onnx"

    torch.onnx.export(
        model,  # model being run
        torch.randint(
            2000, input_shape
        ),  # model input (or a tuple for multiple inputs)
        save_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "emb_linear_onnx",
        onnx_model,
        save_path,
    )
    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.retain_tvm_python_files = True

    verify_module(
        mod,
        (input_shape,),
        input_params=[
            {"data_format": torch.int64},
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    os.remove(save_path)



# input_shapes = [(1, 128, 10, 10)]
# scale_factors = [2, 3]
# upsample_modes = ["nearest", "bilinear"]


# @pytest.mark.parametrize(
#     "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
# )
# @pytest.mark.parametrize(
#     "scale_factors", scale_factors, ids=[f"sfactor({str(s)})" for s in scale_factors]
# )
# @pytest.mark.parametrize(
#     "upsample_mode", upsample_modes, ids=[f"umode({str(u)})" for u in upsample_modes]
# )
# @pytest.mark.parametrize("align_corners", (True, False), ids=["align", "no_align"])
# def test_tvm_upsample2d_onnx(test_kind, test_device, input_shape, scale_factors, upsample_mode, align_corners):
#     if test_kind.is_training():
#         pytest.xfail()  # Backward is currently unsupported

#     _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

#     if align_corners and upsample_mode != "bilinear":
#         pytest.skip()

#     class Upsample2d(nn.Module):
#         def __init__(self, scale_factors, upsample_mode, align_corners):
#             super().__init__()
#             if upsample_mode == "nearest":
#                 self.resize = torch.nn.Upsample(
#                     scale_factor=scale_factors,
#                     mode=upsample_mode,
#                 )
#             else:
#                 self.resize = torch.nn.Upsample(
#                     scale_factor=scale_factors,
#                     mode=upsample_mode,
#                     align_corners=align_corners,
#                 )
#         def forward(self, a):
#             b = self.resize(a)

#             return b

#     model = Upsample2d(scale_factors, upsample_mode, align_corners)
#     save_path = os.path.dirname(os.path.realpath(__file__)) + "/upsample2d_onnx.onnx"

#     torch.onnx.export(model,               # model being run
#                         torch.randn(input_shape),# model input (or a tuple for multiple inputs),
#                         save_path,   # where to save the model (can be a file or file-like object)
#                         export_params=True,        # store the trained parameter weights inside the model file
#                         opset_version=10,          # the ONNX version to export the model to
#                         do_constant_folding=True,  # whether to execute constant folding for optimization
#                         input_names = ['input'],   # the model's input names
#                         output_names = ['output'], # the model's output names
#                         )

#     onnx_model = onnx.load(save_path)
#     onnx.checker.check_model(onnx_model)
#     mod = OnnxModule(
#         "upsample2d_onnx",
#         onnx_model,
#         save_path,
#     )
#     verify_module(
#         mod,
#         (input_shape,),
#         verify_cfg=VerifyConfig(
#             arch=test_device.arch,
#             devtype=test_device.devtype,
#             test_kind=TestKind.INFERENCE,
#         )
#     )
#     os.remove(save_path)
