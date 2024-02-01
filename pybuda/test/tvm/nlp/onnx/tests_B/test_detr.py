# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# DETR basic bring-up tests of tracing functionality
#
import os
import pytest

import onnx
import torch
from transformers import DetrModel

from pybuda import (
    OnnxModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_detr_encoder_layer(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class DeTrEncoderWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.layer = module.encoder.layers[0]
            self.attn_mask = torch.rand((1, 1, 256, 256))
            self.pos_emb = torch.rand((1, 256))

        def forward(self, hidden_states):
            return self.layer(hidden_states, self.attn_mask, self.pos_emb)

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Unsupported HW op: heaviside
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    # Configure PyTorch module
    pytorch_module = DetrModel.from_pretrained(
        "facebook/detr-resnet-50", torchscript=True
    )
    pytorch_module = DeTrEncoderWrapper(pytorch_module)

    # Export to ONNX
    input_shape = (1, 256, 256)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/detr_encoder_layer.onnx"
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "detr_encoder_layer_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors=("layer.self_attn.k_proj.bias"),
        ),
    )

    # Cleanup
    os.remove(save_path)


def test_detr_decoder_layer(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class DeTrDecoderWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.layer = module.decoder.layers[0]
            self.attn_mask = torch.rand((1, 1, 256, 256))
            self.pos_emb = torch.rand((1, 256))
            self.kv_state = torch.rand((1, 1, 256, 256))

        def forward(self, hidden_states):
            return self.layer(
                hidden_states, self.attn_mask, self.pos_emb, self.kv_state
            )

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Unsupported HW op: heaviside
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    # Configure PyTorch module
    pytorch_module = DetrModel.from_pretrained(
        "facebook/detr-resnet-50", torchscript=True
    )
    pytorch_module = DeTrDecoderWrapper(pytorch_module)

    # Export to ONNX
    input_shape = (1, 256, 256)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/detr_decoder_layer.onnx"
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "detr_decoder_layer_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors=("layer.self_attn.k_proj.bias"),
        ),
    )

    # Cleanup
    os.remove(save_path)
