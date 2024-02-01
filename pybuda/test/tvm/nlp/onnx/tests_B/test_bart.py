# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os

import pytest
import torch
from transformers import BartConfig, BartModel
import onnx

from pybuda import (
    OnnxModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_bart_encoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    class BartEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.layer = model.encoder.layers[0]
            self.attn_mask = torch.rand((1, 1, 128, 128))
            self.layer_head_mask = torch.rand((4,))

        def forward(self, hidden_states):
            return self.layer(hidden_states, self.attn_mask, self.layer_head_mask)

    # Configure module
    config = BartConfig.from_pretrained("facebook/bart-base", torchscript=True)
    config.d_model = 128
    config.encoder_layers = 1
    config.decoder_layers = 1
    config.encoder_attention_heads = 4
    config.decoder_attention_heads = 4
    config.encoder_ffn_dim = 128
    config.decoder_ffn_dim = 128
    pytorch_module = BartModel(config)
    pytorch_module = BartEncoderWrapper(pytorch_module)

    # Export to ONNX
    input_shape = (1, 128, 128)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/bart_encoder_layer.onnx"
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
        "bart_encoder_layer_onnx",
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
            waive_gradient_errors=("layer.self_attn.k_proj.bias")
        ),
    )

    # Cleanup
    os.remove(save_path)


def test_bart_decoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    class BartDecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.layer = model.decoder.layers[0]

        def forward(self, hidden_states):
            return self.layer(hidden_states)

    # Configure module
    config = BartConfig.from_pretrained("facebook/bart-base", torchscript=True)
    config.d_model = 128
    config.encoder_layers = 1
    config.decoder_layers = 1
    config.encoder_attention_heads = 4
    config.decoder_attention_heads = 4
    config.encoder_ffn_dim = 128
    config.decoder_ffn_dim = 128
    pytorch_module = BartModel(config)
    pytorch_module = BartDecoderWrapper(pytorch_module)

    # Export to ONNX
    input_shape = (1, 128, 128)
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/bart_decoder_layer.onnx"
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
        "bart_decoder_layer_onnx",
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
        ),
    )

    # Cleanup
    os.remove(save_path)
