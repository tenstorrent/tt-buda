# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BART Demo Script - SQuADv1.1 QA
import pytest
from test.utils import download_model
import os
import torch
from transformers import BartConfig, BartModel, BartTokenizer, BartForSequenceClassification
from transformers.models.bart.modeling_bart import shift_tokens_right, BartAttention

import pybuda
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    CPUDevice,
    TTDevice,
)

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

from loguru import logger


from pybuda.op.eval.common import compare_tensor_to_golden, calculate_pcc

from typing import Optional


def slice_out(model, input_ids, hidden_states):
    hidden_states = hidden_states.float()
    eos_mask = input_ids.eq(model.config.eos_token_id)
    
    if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
        raise ValueError("All examples must have the same number of <eos> tokens.")
    sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
        :, -1, :
    ]
    return sentence_representation

class BartWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        out = self.model(input_ids, attention_mask, decoder_input_ids)[0]
        return out

@pytest.mark.skip(reason="Not supported for release")
def test_pt_bart_classifier(test_device): 
    compiler_cfg = _get_global_compiler_config() 

    model_name = f"facebook/bart-large-mnli"
    model = download_model(BartForSequenceClassification.from_pretrained, model_name, torchscript=True) 
    tokenizer = download_model(BartTokenizer.from_pretrained, model_name, pad_to_max_length=True)
    hypothesis = "Most of Mrinal Sen's work can be found in European collections."
    premise = "Calcutta seems to be the only other production center having any pretensions to artistic creativity at all, but ironically you're actually more likely to see the works of Satyajit Ray or Mrinal Sen shown in Europe or North America than in India itself."
    
    # generate inputs
    inputs_dict = tokenizer( 
        premise,
        hypothesis,
        truncation=True,
        padding='max_length',
        max_length=256,
        truncation_strategy='only_first',
        return_tensors='pt'
    ) 
    decoder_input_ids = shift_tokens_right(inputs_dict["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id)  
    inputs = (inputs_dict["input_ids"], inputs_dict["attention_mask"], decoder_input_ids) 

    # Compile & feed data
    pt_mod = BartWrapper(model.model)
    mod = PyTorchModule("bart", pt_mod)
    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=mod)
    tt0.push_to_inputs(inputs)
    output_q = pybuda.run_inference()

    # Verify output
    outputs = output_q.get()[0].value().detach().float()
    fw_outputs = pt_mod(*inputs)[0] 
    outputs = slice_out(model, inputs[0], outputs)
    fw_outputs = slice_out(model, inputs[0], fw_outputs.unsqueeze(0)) 
    assert compare_tensor_to_golden("bart_sliced_out", fw_outputs, outputs, pcc=0.99)
 
