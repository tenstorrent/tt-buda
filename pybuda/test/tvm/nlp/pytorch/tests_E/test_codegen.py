# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from transformers import WhisperConfig, WhisperModel, AutoProcessor, AutoFeatureExtractor, WhisperProcessor

import pytest

import torch
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
)

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model

from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, CodeGenForCausalLM


def test_codegen_single_layer_fallback(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.retain_tvm_python_files = True

    framework_model = download_model(CodeGenForCausalLM.from_pretrained, "Salesforce/codegen-350M-mono", use_cache=False, n_layer=1, return_dict=False)

    class CodegenTransformer(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, ):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
            return outputs
    mod = PyTorchModule("CodegenTransformer", CodegenTransformer(framework_model.transformer))
    input_shape = (1, 5)
    input_ids = torch.tensor([[ 4299, 23748,    62,  6894, 33529]])
    input_shape2 = (1, 5)
    attention_mask = torch.ones(input_shape2)

    verify_module(
        mod,
        (input_shape, input_shape2,),
        inputs=[(input_ids, attention_mask,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_pybuda_codegen_vs_framework=True,
            run_golden=True,
            pcc=0.98
        ),
        input_params=[
            {"requires_grad": False, "data_format": torch.int},
            {"requires_grad": False, "data_format": torch.int},
        ],
    )

