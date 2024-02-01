# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.verify.backend import verify_module
import pytest

import torch
import torch.nn as nn

import pybuda
import pybuda.op

from pybuda import (
    TTDevice,
    BackendType,
    pybuda_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
)
from pybuda.config import CompileDepth

from pybuda import TTDevice, VerifyConfig, pybuda_compile


# from .pytorch_unet import UNetModel
from test.legacy_tests.clip_guided_diffusion.clip.clip_torch import CLIP, VisionTransformer, create_CLIP
from pybuda.verify import verify_module_pipeline
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import _get_global_compiler_config

def test_tvm_CLIP(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    pytorch_clip_text_encoder, pytorch_clip_without_text_encoder, pytorch_clip, clip_config = create_CLIP()
    encoded_text = pytorch_clip_text_encoder()
    encoded_text_shape = (1, *encoded_text.shape[1:])

    clip_mod = PyTorchModule("CLIP", pytorch_clip_without_text_encoder)
    image_shape = (1, 3, 224, 224)
    image = torch.rand(image_shape)

    verify_module(
        clip_mod,
        (image_shape, encoded_text_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


def test_tvm_visiontransformer(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    if test_kind.is_training():
        pytest.skip()

    num_heads = 12
    image_resolution = 224
    vision_layers = 1 #12
    vision_width = 768
    vision_patch_size = 32
    embed_dim = 512
    scale = vision_width ** -0.5

    class_embedding = torch.randn(vision_width)
    rand_resolution = torch.randn((image_resolution // vision_patch_size) ** 2 + 1, vision_width)
    proj = torch.randn(vision_width, embed_dim)

    model = VisionTransformer(
        input_resolution=image_resolution, layers=vision_layers, 
        width=vision_width, patch_size=vision_patch_size, 
        heads=num_heads, output_dim=embed_dim, 
        class_embedding=class_embedding, proj=proj,
        rand_resolution=rand_resolution
    )
    # batch_size = 16, but for now targeting 1 until verify_module has support for batch_size > 1
    image = torch.rand((1, 3, 224, 224)) # actual input
    
    pytorch_out = model(image)

    mod = PyTorchModule("VisionTransformer", model)
    verify_module(
        mod,
        ((1, 3, 224, 224),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )
