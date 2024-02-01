# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

import os

import pybuda
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_squeezebert_sequence_classification_pytorch(test_device):
    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, "squeezebert/squeezebert-mnli")
    model = download_model(AutoModelForSequenceClassification.from_pretrained,
        "squeezebert/squeezebert-mnli"
    )

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Example from multi-nli validation set
    text = """Hello, my dog is cute"""

    # Data preprocessing
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_bart", model),
        input_shapes=[(input_tokens.shape,)],
        inputs=[(input_tokens,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
