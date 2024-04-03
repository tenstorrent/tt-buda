# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt2_mq import GPT2LMHeadCustomModel as GPT2MQModel

from pybudify import PyBudify
#from monkeypatch import monkeypatch

import pytest
import pybuda
from pybuda.config import _get_global_compiler_config

# Pytest to run santacoder model
@pytest.mark.parametrize("tokens", [10, 100])
@pytest.mark.parametrize("device", ["silicon"])
@pytest.mark.parametrize("arch", ["greyskull", "wormhole_b0"])
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16", "fp8", "fp8b"])
@pytest.mark.parametrize("amp_level", ["amp0", "amp1", "amp2"])
@pytest.mark.parametrize("num_chips", ["chip1", "chip2", "chip32"])
@pytest.mark.parametrize("fuse", ["fuse", "no_fuse"])
def test_santacoder(tokens, device, arch, precision, amp_level, num_chips, fuse):
    pybuda.config.set_configuration_options(default_df_override=pybuda.DataFormat.Float16)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.amp_properties.append(pybuda._C.AMPNodeProperties(
        op_type="splice",
        output_df=pybuda._C.DataFormat.Float16,
        accumulate_df=pybuda._C.DataFormat.Float16,
        math_fidelity=pybuda.MathFidelity.HiFi3,
        intermediate_df=pybuda._C.DataFormat.Float16,
        input_df= {0: [pybuda._C.DataFormat.Float16, True], 1: [pybuda._C.DataFormat.Float16, True]}
    ))


    ground_truth = " print(\"Hello World!\")\n\n"

    if amp_level == "amp0":
        amp_level = 0
    elif amp_level == "amp1":
        amp_level = 1
    elif amp_level == "amp2":
        amp_level = 2

    if num_chips == "chip1":
        num_chips = 1
    elif num_chips == "chip2":
        num_chips = 2
    elif num_chips == "chip32":
        num_chips = 32

    if fuse == "fuse":
        fuse = True
    else:
        fuse = False

    # Construct parameters object
    parameters = {
        'model': 'bigcode/santacoder',
        'kv_cache': 'pybuda/test/santacoder/kv_cache.pt',
        'stop': '\n\n',
        'num_tokens': tokens,
        'output_at_end': False,
        'device': device,
        'arch': arch,
        'precision': precision,
        'amp_level': amp_level,
        'num_chips': 1,
        'fuse': fuse,
        'perf': None,
        'verify': False,
        'log_level': 'ERROR',
        'load': None,
        'save': None}

    result = main(parameters)
    print("|" + result + "|")

    assert(result == ground_truth)

def main(parameters):
    # Download the model and tokenizer
    model = GPT2MQModel.from_pretrained(parameters["model"])
    tokenizer = AutoTokenizer.from_pretrained(parameters["model"])
    model.eval()

    # Initialize the past_key_values with zeros
    past_key = torch.zeros((1, len(model.transformer.h), 128, 2047), dtype=torch.float32)
    past_value = torch.zeros((1, len(model.transformer.h), 2047, 128), dtype=torch.float32)

    # Load the KV cache
    input_ids, past_key_values = torch.load(parameters["kv_cache"])
    num_tokens = past_key_values[0][0].shape[-1]

    # past_key_values is a tuple of (key, value) tensors, one for each layer
    # copy into the right parts of the key, value tensors
    for i, (k, v) in enumerate(past_key_values):
        past_key[0, i, :, :num_tokens] = k
        past_value[0, i, :num_tokens, :] = v

    all_text = ''
    with torch.no_grad():
        # Now transition to running in token-by-token mode for generation
        new_tokens = 0

        if parameters["device"] != 'huggingface':
            model.transformer.blocks = PyBudify(model.transformer.blocks, device=parameters["device"], arch=parameters["arch"], precision=parameters["precision"], amp_level=parameters["amp_level"],
                num_chips=parameters["num_chips"], fuse=parameters["fuse"], perf=parameters["perf"], verify=parameters["verify"], log_level=parameters["log_level"], tti_load=parameters["load"], tti_save=parameters["save"])

        while True:
            if parameters["stop"] and parameters["stop"] in all_text:
                break

            if parameters["num_tokens"] and new_tokens >= parameters["num_tokens"]:
                break

            # Generate the next token. We track position_id to get correct positional embeddings.
            position_ids = torch.tensor([[num_tokens]], dtype=torch.long, device=input_ids.device)
            attention_mask = torch.ones((1, 2048), dtype=torch.long, device=input_ids.device)
            attention_mask[:, num_tokens:-1] = 0 # Mask out unused tokens; current token is appended to cache in attention computation
            output, new_key, new_value = model(input_ids, past_key, past_value, attention_mask, position_ids)
            token = output[-1].argmax(dim=-1)
            input_ids = token.unsqueeze(0)

            # Insert the new key, value into the past_key, past_value tensors at the right position
            past_key[:, :, :, num_tokens] = new_key[:, :, :, 0]
            past_value[:, :, num_tokens, :] = new_value[:, :, 0, :]

            # Update the new and total token counts
            new_tokens += 1
            num_tokens += 1

            # Print the generated token
            text = tokenizer.decode(token, clean_up_tokenization_spaces=True)
            if not parameters["output_at_end"]:
                print(text, end='', flush=True)
            all_text += text

        if parameters["output_at_end"]:
            print(all_text)

    return all_text


if __name__ == '__main__':
    parser = ArgumentParser('Generate text token-by-token starting with a pre-filled KV cache')
    parser.add_argument('-m', '--model', type=str, default='bigcode/santacoder', help='Model name')
    parser.add_argument('-k', '--kv-cache', type=str, default='kv_cache.pt', help='KV cache file')
    parser.add_argument('-s', '--stop', type=str, default='\n\n', help='Text to stop decoding after')
    parser.add_argument('-n', '--num-tokens', type=int, default=10, help='Maximum number of tokens to generate')
    parser.add_argument('--output-at-end', action='store_true', help='Output at the end of generation instead of token by token')

    parser.add_argument('-d', '--device', choices=['huggingface', 'pytorch', 'golden', 'silicon'], default='huggingface', help='huggingface: run using HF code only, pytorch: use our shim but run in PyTorch, golden/silicon: run via pybuda')
    parser.add_argument('--arch', choices=['greyskull', 'wormhole_b0'], default='wormhole_b0', help='Architecture to use for silicon')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16', 'fp8', 'fp8b'], default='fp32', help='Precision to use for all silicon tensors')
    parser.add_argument('--amp-level', type=int, choices=[0, 1, 2], help='Automatic mixed precision level (0=off, 1=mixed b-formats, 2=mixed a-formats)')
    parser.add_argument('--num-chips', type=int, default=1, help='Number of chips to use')
    parser.add_argument('--fuse', action='store_true', help='Fuse layers')
    parser.add_argument('--perf', choices=['none', 'light', 'verbose'], default=None, help='Performance tracing')
    parser.add_argument('--verify', action='store_true', help='Verify results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='ERROR', help='Log level')
    parser.add_argument('--load', type=str, help='Load a TTImage')
    parser.add_argument('--save', type=str, help='Save a TTImage')
    args = parser.parse_args()

    main(vars(args))
