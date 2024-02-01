# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Falcon-7B Decode tests

import os
import json
import pytest
import torch
from typing import Dict

from transformers import AutoTokenizer

from .utils import get_tokenizer, make_data_module
from ..pybudify import PyBudify
from ..models.falcon7b.tt_modeling_RW_pad import RWForCausalLM as RWForCausalLMPadded
from ..models.falcon7b.tt_modeling_RW_pad_masked_odkv import RWForCausalLM as RWForCausalLMPaddedMaskedODKV
from ..models.falcon7b.configuration_RW import RWConfig

from .falcon_modules.falcon import run_sync, run_demo_sync_finetune, run_demo_sync_masked_odkv

# Requires around 14 GB for falcon7B model (tiiuae/falcon-7b)
os.environ['HF_DATASETS_CACHE']='/proj_sw/large-model-cache/falcon7b'
os.environ['TRANSFORMERS_CACHE']='/proj_sw/large-model-cache/falcon7b'

# Tests Falcon-7B Decode demo: padded version, single user
# This is similar to running the following command:
# python decode.py --mode sequential --stop="" --num-tokens 100 --layers 32 --seqlen 2048 -d silicon 
# --arch wormhole_b0 --precision bf16 --kv-cache data/two_cities_kv.pt --version padded --batch-size 1 
# --matmul-precision weight_bfp8_act_bf16
@pytest.mark.parametrize("num_layers,sequence_length", [(32,2048)])
def test_decode_padded(num_layers, sequence_length):
    n_layers = num_layers
    seqlen = sequence_length

    # using padded model
    with open("pybuda/test/falcon/models/falcon7b/config_padded.json", "r") as f:
        params = json.loads(f.read())
    config = RWConfig(**params, user_rows=1)
    config.n_layer = n_layers
    model = RWForCausalLMPadded.from_pretrained('tiiuae/falcon-7b', config=config)#, local_files_only=True)
    model.transformer.split_qkv_weights()
    model.transformer.pad_decoders() # After loading weights, pad the decoders
       
    tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')

    model.eval()
    dh = model.transformer.h[0].self_attention.head_dim

    # Load the KV cache
    input_ids, past_key_values = torch.load('third_party/confidential_customer_models/model_3/two_cities_kv.pt')
    num_tokens = past_key_values[0][0].shape[-2] # [b x s x dh]

    # limit the layers in past_key_values
    past_key_values = past_key_values[:n_layers]

    ext_past_kv = []
    # past_key_values is a tuple of [key, value] tensors, one for each layer
    # copy into the right parts of the key, value tensors
    for i, (k, v) in enumerate(past_key_values):
        past_key = torch.zeros((1, seqlen - 1, dh), dtype=torch.float32)
        past_value = torch.zeros((1, seqlen - 1, dh), dtype=torch.float32)

        past_key[0, :num_tokens, :] = k
        past_value[0, :num_tokens, :] = v

        ext_past_kv.append([past_key, past_value])

    past_key_values = tuple(ext_past_kv)

    with torch.no_grad():
        model.transformer.blocks = PyBudify(
            model.transformer.blocks,
            precision='bf16',
            matmuls = 'weight_bfp8_act_bf16',
            netlist_name = 'falcon7b_decode_padded',
            )

        generated_output = run_sync(model, tokenizer, input_ids, past_key_values, num_tokens, seqlen, n_layers)
        expected_output = ' we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Hell, we were all going'
        expected_output2 = ' we were all going direct to Hell, we were all going direct to Hell, we were all going direct to Heaven, we were all going direct to Hell, we were all going direct to Heaven, we were all going direct to Hell, we were all going direct to Heaven, we were all going direct to Hell, we were all going direct to Heaven, we were all going direct to Hell, we were all going direct to Heaven, we were all going direct to Hell, we were all going'
        print(f"Generated output={generated_output}")
        assert expected_output in generated_output or expected_output2 in generated_output, \
                "Neither of the expected outputs is contained within generated output"


# Tests Falcon-7B Decode demo: masked_odkv version, 32 users (checks if all users match)
# This is similar to running the following command:
# python -u decode_demo.py --max-tokens 120 --user-rows 32 --seqlen 512 
# --device silicon --precision bf16 --prompts-file data/two_cities.json
# --output-at-end --matmul-precision weight_bfp8_act_bf16
@pytest.mark.parametrize("num_layers,sequence_length,prompts_file", [(32,512,'pybuda/test/falcon/data/two_cities.json')])
def test_decode_demo_masked_odkv(num_layers, sequence_length,prompts_file):
    user_rows = 32

    # using padded model
    with open("pybuda/test/falcon/models/falcon7b/config_padded.json", "r") as f:
        params = json.loads(f.read())
    config = RWConfig(**params, user_rows=user_rows)
    config.n_layer = num_layers
    model = RWForCausalLMPaddedMaskedODKV.from_pretrained('tiiuae/falcon-7b', config=config)
    model.transformer.split_qkv_weights()
    model.transformer.pad_decoders() # After loading weights, pad the decoders

    tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')
    tokenizer.pad_token_id = 0 #'[PAD]'     # set pad token for tokenizer

    prompts = json.load(open(prompts_file))
    tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
    
    model.eval()

    with torch.no_grad():
        # Now transition to running in token-by-token mode for generation
        model.transformer.blocks = PyBudify(
            model.transformer.blocks,
            precision='bf16',
            masked_odkv=True,
            num_layers=num_layers,
            matmuls = 'weight_bfp8_act_bf16',
            netlist_name = 'falcon7b_decode_demo_masked_odkv',
            )
        
        generated_output = run_demo_sync_masked_odkv(model, tokenizer, tokenized, num_layers, sequence_length)
        expected_output = ', we were all going direct to Hell, we were all going direct to Hell,'
        print(f"Generated output={generated_output}")
        validated = 0
        for user_output in generated_output:
            if expected_output in user_output:
                validated += 1

        assert validated == 32, f"{32-validated} out of 32 users don't contain the expected output"


# Tests full finetuned decode demo - Validates perplexity
# This is similar to running the following command:
# python decode_demo.py --mode sequential --max-tokens 256 --layers 32
# --seqlen 512 -d silicon --arch wormhole_b0 --precision bf16 --version finetune
# --batch-size 1 --user-rows 32  --dataset alpaca_eval --output-at-end
# --output-eval -t 1.0 --matmul-precision weight_bfp8_act_bf16 --perplexity --ci
@pytest.mark.parametrize("num_layers,sequence_length", [(32,512)])
def test_finetune_decode_ppl(num_layers, sequence_length):
    user_rows = 32

    tokenizer = get_tokenizer('tiiuae/falcon-7b', explicit_pad_token=False)
    #using padded model
    with open("pybuda/test/falcon/models/falcon7b/config_padded.json", "r") as f:
        params = json.loads(f.read())
    config = RWConfig(**params, user_rows=user_rows)
    config.n_layer = num_layers

    config.bos_token_id=tokenizer.bos_token_id
    config.eos_token_id=tokenizer.eos_token_id
    config.pad_token_id=tokenizer.pad_token_id

    model = RWForCausalLMPaddedMaskedODKV.from_pretrained('tiiuae/falcon-7b', config=config)
    model.transformer.split_qkv_weights()
    model.transformer.pad_decoders() # After loading weights, pad the decoders

    # if args.adapter:
    #     adapters_name = args.adapter
    #     model = PeftModel.from_pretrained(model, adapters_name, cache_dir=args.hf_cache)
    #     print(f"Merging adapter")
    #     model = model.merge_and_unload() #model ready to prefill and decode

    data_module = make_data_module(tokenizer, 'alpaca_eval', train=False, eval=True, max_train_samples=None, max_eval_samples=None,
                                    max_seq_len=sequence_length, data_seed=42, filter_longer_sequences=False, filter_longer_inputs=True)
    eval_dataset = data_module["eval_dataset"]
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=user_rows, shuffle=False, drop_last=False)

    model.eval()

    with torch.no_grad():
        model.transformer.blocks = PyBudify(
            model.transformer.blocks,
            precision='bf16',
            masked_odkv=True,
            num_layers=num_layers,
            matmuls = 'weight_bfp8_act_bf16',
            netlist_name = 'falcon7b_finetune_decode_ppl',
            )

        run_demo_sync_finetune(model, tokenizer, dataloader, num_layers, sequence_length)
