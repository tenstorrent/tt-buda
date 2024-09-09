# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model

import os

import pybuda
from transformers import BertForMaskedLM, BertTokenizer, BertForTokenClassification, BertForSequenceClassification, BertForQuestionAnswering

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, NebulaGalaxy

def generate_model_bert_maskedlm_hf_pytorch(test_device, variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    model = BertForMaskedLM.from_pretrained(model_ckpt)

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

    model = pybuda.PyTorchModule("pt_bert_masked_lm", model)

    return model, [input_tokens['input_ids']], {}

def test_bert_masked_lm_pytorch(test_device):
    model, inputs, _ = generate_model_bert_maskedlm_hf_pytorch(
        test_device, "bert-base-uncased",
    )

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )


def generate_model_bert_qa_hf_pytorch(test_device, variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = download_model(BertTokenizer.from_pretrained, model_ckpt)
    model = download_model(BertForQuestionAnswering.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

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
    
    model = pybuda.PyTorchModule("pt_bert_question_answering", model)
    
    return model, [input_tokens['input_ids']], {}

    
def test_bert_question_answering_pytorch(test_device):
    model, inputs, _ = generate_model_bert_qa_hf_pytorch(
        test_device, "bert-large-cased-whole-word-masking-finetuned-squad",
    )

    if test_device.arch == BackendDevice.Blackhole:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{42*1024}"

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )


def generate_model_bert_seqcls_hf_pytorch(test_device, variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = download_model(BertTokenizer.from_pretrained, model_ckpt)
    model = download_model(BertForSequenceClassification.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

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
    
    model = pybuda.PyTorchModule("pt_bert_sequence_classification", model)
    
    return model, [input_tokens['input_ids']], {}


def test_bert_sequence_classification_pytorch(test_device):
    model, inputs, _ = generate_model_bert_seqcls_hf_pytorch(
        test_device, "textattack/bert-base-uncased-SST-2",
    )

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=False,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
    
def generate_model_bert_tkcls_hf_pytorch(test_device, variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = download_model(BertTokenizer.from_pretrained, model_ckpt)
    model = download_model(BertForTokenClassification.from_pretrained, model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object 
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

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
    
    model = pybuda.PyTorchModule("pt_bert_token_classification", model)
    
    return model, [input_tokens['input_ids']], {}


def test_bert_token_classification_pytorch(test_device):
    model, inputs, _ = generate_model_bert_tkcls_hf_pytorch(
        test_device, "dbmdz/bert-large-cased-finetuned-conll03-english",
    )

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
        )
    )
