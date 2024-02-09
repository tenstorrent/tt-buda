# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, DataFormat, NebulaGalaxy

import os
import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, OPTForQuestionAnswering, OPTForSequenceClassification

variants = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_opt_causal_lm(variant, test_device):
    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = DataFormat.Float16_b
    if variant == "facebook/opt-1.3b":
        compiler_cfg.amp_level = 2

        # Disable expanding output buffer of fork nodes - causes out of memory issue in blobgen.
        os.environ["PYBUDA_FORK_JOIN_EXPAND_FORK_OUTPUT_BUF"] = "0"
    if variant == "facebook/opt-350m":
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    config = OPTConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict['return_dict'] = False
    config_dict['use_cache'] = False
    config = OPTConfig(**config_dict)
    model = download_model(OPTForCausalLM.from_pretrained, variant, config=config)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"
    input_tokens = tokenizer(
        prefix_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_opt_causal_lm", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.7,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )

@pytest.mark.parametrize("variant", variants, ids=variants)
def test_opt_qa(variant, test_device):
    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = DataFormat.Float16_b
    if variant == "facebook/opt-1.3b":
        compiler_cfg.default_df_override = DataFormat.Float16

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(OPTForQuestionAnswering.from_pretrained,
        variant, torchscript=True
    )

    # Load data sample
    question, context = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_opt_question_answering", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.7,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )

@pytest.mark.parametrize("variant", variants, ids=variants)
def test_opt_sequence_classification(variant, test_device):
    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    compiler_cfg.default_df_override = DataFormat.Float16_b
    if variant == "facebook/opt-1.3b" or variant == "facebook/opt-350m":
        compiler_cfg.enable_auto_fusing = False

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(OPTForSequenceClassification.from_pretrained,
        variant, torchscript=True
    )

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_opt_sequence_classification", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.93,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
