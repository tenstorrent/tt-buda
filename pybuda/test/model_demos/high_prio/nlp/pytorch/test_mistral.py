# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig

import pybuda
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendDevice
from pybuda._C import DataFormat, MathFidelity
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda.transformers.pipeline import NLPPipelineWrapper


variants = ['mistralai/Mistral-7B-v0.1']
@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mistral_decoder_layer(variant, test_device):

    if test_device.arch != BackendDevice.Wormhole_B0:
        pytest.skip("Currently only supported on Wormhole B0 N150 device")

    model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto")
    model.eval()
    module = model.model.layers[0]

    # test should work for batch size 1 and seqlen <= 128
    # for larger seqlen, a problem with valid node placement can occur
    batch_size = 1
    hidden_dim = 4096
    seqlen = 128

    sample_inputs = torch.randn(batch_size, seqlen, hidden_dim)

    verify_module(
        pybuda.PyTorchModule(
            f"mistral_decoder_layer_seqlen_{seqlen}_bs_{batch_size}", module),
            input_shapes=[(sample_inputs.shape,)],
            inputs=[(sample_inputs,)],
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                devmode=test_device.devmode,
                test_kind=TestKind.INFERENCE,
        )
    )


variants = ['mistralai/Mistral-7B-v0.1']
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mistral(variant, test_device):
    if test_device.arch != BackendDevice.Wormhole_B0:
        pytest.skip("Currently only supported on Wormhole B0 N150 device")

    configuration = MistralConfig()

    configuration.sliding_window = None
    configuration.use_cache = False
    configuration.return_dict = False

    pybuda.set_configuration_options(default_df_override=pybuda.DataFormat.Float16_b)

    # configuration for all ops that are not matmul
    pybuda.config.configure_mixed_precision(
        op_type='^((?!matmul).)*$',
        math_fidelity=MathFidelity.HiFi4,
        accumulate_df=DataFormat.Float16_b
    )

    # configuration for all matmul ops
    # when inputs to matmuls are Bfp8_b, the whole model can fit to single chip
    pybuda.config.configure_mixed_precision(
        op_type='matmul',
        math_fidelity=MathFidelity.HiFi4,
        input_df={0:[DataFormat.Bfp8_b, False], 1:[DataFormat.Bfp8_b, False]},
        accumulate_df=DataFormat.Float16_b
    )

    module = AutoModelForCausalLM.from_pretrained(variant, device_map="auto", config = configuration)
    tokenizer = AutoTokenizer.from_pretrained(variant)
    
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

    # test should work for batch size 1 and seqlen <= 128
    # for larger seqlen, a DRAM allocation problem might occur (this model is already near maximum model size for single chip)
    batch_size = 1
    prompt = "Of course, fancy writing doesn't just conceal ideas. It can also conceal the lack of them. That's why some people write that way, to conceal the fact that they have nothing to say. Whereas writing simply keeps you honest. If you say nothing simply, it will be obvious to everyone, including you. Simple writing also lasts better. People reading your stuff in the future will be in much the same position as people from other countries reading it today. The culture and the language will have changed. It's not vain to care about that, any more than it's vain for "
    sample_inputs = tokenizer(prompt, return_tensors = 'pt')['input_ids']

    verify_module(
        pybuda.PyTorchModule(
            f"full_model_seqlen_{sample_inputs.shape[-1]}_bs_{batch_size}_layers_{configuration.num_hidden_layers}", module),
            input_shapes=[(sample_inputs.shape,)],
            inputs=[(sample_inputs, )],
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                devmode=test_device.devmode,
                test_kind=TestKind.INFERENCE,
        )
    )

variants = ['mistralai/Mistral-7B-v0.1']
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="This test currently serves the same purpose as test_mistral")
def test_mistral_decode(variant, test_device):
    if test_device.arch != BackendDevice.Wormhole_B0:
        pytest.skip("Currently only supported on Wormhole B0 N150 device")

    configuration = MistralConfig()
    configuration.sliding_window = None
    configuration.use_cache = False
    configuration.return_dict = False

    pybuda.set_configuration_options(default_df_override=pybuda.DataFormat.Float16_b)

    # configuration for all ops that are not matmul
    pybuda.config.configure_mixed_precision(
        op_type='^((?!matmul).)*$',
        math_fidelity=MathFidelity.HiFi4,
        accumulate_df=DataFormat.Float16_b
    )

    # configuration for all matmul ops
    # when inputs to matmuls are Bfp8_b, the whole model can fit to single chip
    pybuda.config.configure_mixed_precision(
        op_type='matmul',
        math_fidelity=MathFidelity.HiFi4,
        input_df={0:[DataFormat.Bfp8_b, False], 1:[DataFormat.Bfp8_b, False]},
        accumulate_df=DataFormat.Float16_b
    )

    pytorch_model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto", config = configuration)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    pytorch_model.eval()
    for param in pytorch_model.parameters():
        param.requires_grad = False
    
    tokenizer.pad_token = tokenizer.eos_token

    prompt = 'I am a dwarf and '
    inputs = tokenizer(prompt, return_tensors="pt")

    max_generated_tokens = 100

    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=max_generated_tokens)
    generated_pt_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nPyTorch (sanity) generated:")
    pt_ans = generated_pt_text.split('\n\n')
    print(f"{pt_ans}")

    wrapper = NLPPipelineWrapper(
        pytorch_model,
        tokenizer,
        pytorch_model.__class__.__name__,
        use_cache=None,
        forward_fn=None,
        max_length=max_generated_tokens
        )
    
    pytorch_model.prepare_inputs_for_generation = wrapper.prepare_inputs_for_generation

    # this generates sample text, to trigger model compilation, so it is not factored during latency measurement
    outputs = pytorch_model.generate(inputs['input_ids'][:,0:1], do_sample=False, max_length=max_generated_tokens)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    start = time.time()
    outputs = pytorch_model.generate(inputs['input_ids'], do_sample=False, max_length=max_generated_tokens)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    end = time.time()

    num_generated_tokens = outputs.shape[-1] - inputs['input_ids'].shape[-1]
    print('TT generated:')
    print(output_text[0])
    print(f'Tokens / s: {num_generated_tokens / (end-start)}')
