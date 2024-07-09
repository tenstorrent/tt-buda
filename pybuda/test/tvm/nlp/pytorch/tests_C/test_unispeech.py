# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# UniSpeech basic bring-up tests of tracing functionality
#
import pytest

import torch
from transformers import UniSpeechModel

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

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model

def test_unispeech_feature_projection(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    framework_model = download_model(
        UniSpeechModel.from_pretrained,
        "microsoft/unispeech-sat-base", torchscript=True
    )

    framework_submodel = framework_model.feature_projection
    mod = PyTorchModule(
        "unispeech_feature_projection",
        framework_submodel,
    )

    input_shape = (1, 512)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_unispeech_transformer_encoder(training):
    #TODO: Convert this into verify_module flow
    recompute = True  # Always run with recompute in post-commit CI. Nightly tests both

    if training:
        compile_depth = CompileDepth.PRE_LOWERING_PASS  # Data mismatch
    else:
        compile_depth = CompileDepth.PRE_LOWERING_PASS  # Unsupported HW ops

    framework_model = download_model(
        UniSpeechModel.from_pretrained,
        "microsoft/unispeech-sat-base", torchscript=True
    )

    framework_submodel = framework_model.encoder.layers[0]
    module = PyTorchModule(
        "unispeech_transformer_encoder",
        framework_submodel,
    )

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    input_shape = (48, 128, 768)
    inputs = [
        torch.rand(input_shape),
    ]

    pybuda_model_results = pybuda_compile(
        tt0,
        "unispeech_transformer_encoder",
        *inputs,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
            waive_gradient_errors={
                "attention.k_proj.bias",  # Too small values
                "layer_norm.weight",
                "layer_norm.bias",
                "feed_forward.intermediate_dense.weight",
                "feed_forward.intermediate_dense.bias",
            },
        ),
    )

    evaluate_framework_vs_pybuda(framework_submodel, pybuda_model_results, *inputs)
