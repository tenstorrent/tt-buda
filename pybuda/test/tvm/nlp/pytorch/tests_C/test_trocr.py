# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# TrOCR basic bring-up tests of tracing functionality
#
import torch
import pytest
from transformers import (
    TrOCRConfig,
    TrOCRForCausalLM,
    ViTConfig,
    ViTModel,
    VisionEncoderDecoderModel,
)

from pybuda import PyTorchModule, VerifyConfig
from pybuda.verify import verify_module
from pybuda.config import _get_global_compiler_config


def test_trocr_reduced_size(test_kind, test_device):
    # import os
    # os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    # os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"
    
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model.generate(pixel_values)

    # Compile configuration
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.enable_tm_cpu_fallback = False
    compiler_cfg.balancer_policy = "Ribbon"

    # Input shape
    input_shape = (1, 3, 128, 128)

    # Load model
    encoder_config = ViTConfig()
    encoder_config.num_attention_heads = 1
    encoder_config.num_hidden_layers = 1
    encoder_config.image_size = input_shape[-1]
    encoder = ViTModel(encoder_config)

    decoder_config = TrOCRConfig()
    decoder_config.decoder_attention_heads = 1
    decoder_config.decoder_layers = 1
    decoder = TrOCRForCausalLM(decoder_config)
    framework_model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    framework_model = Module(framework_model)

    # Larger variation - also works
    # # Input shape
    # input_shape = (1, 3, 128, 128)

    # # Load model
    # encoder_config = ViTConfig()
    # encoder_config.num_attention_heads = 4
    # encoder_config.num_hidden_layers = 4
    # encoder_config.image_size = input_shape[-1]
    # encoder = ViTModel(encoder_config)

    # decoder_config = TrOCRConfig()
    # decoder_config.decoder_attention_heads = 4
    # decoder_config.decoder_layers = 4
    # decoder = TrOCRForCausalLM(decoder_config)
    # framework_model =  VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    # framework_model = Module(framework_model)

    # Sanity check
    # pixel_values = torch.rand(input_shape)
    # generated_ids = framework_model(pixel_values)

    verify_module(
        PyTorchModule("pt_trocr", framework_model),
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            # verify_all=True,
        ),
    )
