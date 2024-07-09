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
