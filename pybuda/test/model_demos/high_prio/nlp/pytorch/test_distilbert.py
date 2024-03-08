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
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertForTokenClassification, DistilBertForSequenceClassification

variants = ["distilbert-base-uncased", "distilbert-base-cased", "distilbert-base-multilingual-cased"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_distilbert_masked_lm_pytorch(variant, test_device):
    # Load DistilBert tokenizer and model from HuggingFace
    # Variants: distilbert-base-uncased, distilbert-base-cased,
    # distilbert-base-multilingual-cased
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_ckpt = "distilbert-base-uncased"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    model = download_model(DistilBertForMaskedLM.from_pretrained, variant)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    
    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_distilbert_masked_lm", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def test_distilbert_question_answering_pytorch(test_device):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = "distilbert-base-cased-distilled-squad"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, model_ckpt)
    model = download_model(DistilBertForQuestionAnswering.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample from SQuADv1.1
    context = """Super Bowl 50 was an American football game to determine the champion of the National Football League
    (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
    National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.
    The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
    As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed
    initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
    (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently
    feature the Arabic numerals 50."""

    question = "Which NFL team represented the AFC at Super Bowl 50?"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_distilbert_question_answering", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape)],
        inputs=[(input_tokens['input_ids'],input_tokens['attention_mask'])],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def test_distilbert_sequence_classification_pytorch(test_device):

    # Load DistilBert tokenizer and model from HuggingFace
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, model_ckpt)
    model = download_model(DistilBertForSequenceClassification.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    # Temporary disabling t-streaming for distilbert
    compiler_cfg.enable_t_streaming = False

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        pybuda.PyTorchModule("pt_distilbert_sequence_classification", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def test_distilbert_token_classification_pytorch(test_device):
    # Load DistilBERT tokenizer and model from HuggingFace
    model_ckpt = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, model_ckpt)
    model = download_model(DistilBertForTokenClassification.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    # Temporary disabling t-streaming for distilbert
    compiler_cfg.enable_t_streaming = False

    # Load data sample
    sample_text = "HuggingFace is a company based in Paris and New York"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    pcc = 0.98 if test_device.devtype == BackendType.Silicon else 0.99

    verify_module(
        pybuda.PyTorchModule("pt_distilbert_token_classification", model),
        input_shapes=[(input_tokens['input_ids'].shape, input_tokens['attention_mask'].shape,)],
        inputs=[(input_tokens['input_ids'], input_tokens['attention_mask'],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
