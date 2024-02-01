# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Wav2Vec2 basic bring-up tests of tracing functionality
#
import pytest

import torch
from transformers import Wav2Vec2Model

from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model


def test_wav2vec2(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    pytest.skip() # See tenstorrent/pybuda#1935
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"encoder.pos_conv_embed.conv.weight_v"}
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH  

    framework_model = download_model(
        Wav2Vec2Model.from_pretrained,
        "facebook/wav2vec2-base", torchscript=True
    )

    mod = PyTorchModule(
        "wav2vec2_full_model",
        framework_model,
    )

    input_shape = (1, 512)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        )
    )



def test_wav2vec2_base_conv_feature_encoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    framework_model = download_model(
        Wav2Vec2Model.from_pretrained,
        "facebook/wav2vec2-base", torchscript=True
    )

    framework_submodel = framework_model.feature_extractor.conv_layers[0]
    mod = PyTorchModule(
        "wav2vec2_conv_feature_encoder",
        framework_submodel,
    )

    input_shape = (1, 1, 512)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

# TODO: Increase batch dim when possible
def test_wav2vec2_base_transformer_encoder(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()

    framework_model = download_model(
        Wav2Vec2Model.from_pretrained,
        "facebook/wav2vec2-base", torchscript=True
    )

    framework_submodel = framework_model.encoder.layers[0]
    module = PyTorchModule(
        "wav2vec2_transformer_encoder",
        framework_submodel,
    )

    input_shape = (1, 128, 768)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={'attention.k_proj.bias'}
        )
    )
