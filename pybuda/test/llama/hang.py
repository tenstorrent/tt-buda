# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
'''
This script is used to evaluate the performance of the TT model on a dataset
'''

from argparse import ArgumentParser
import torch
from modeling_alpaca_caching import AlpacaForCausalLM
from transformers import LlamaConfig, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import time

from pybudify_caching import PyBudify


def main():

    parser = ArgumentParser('Generate text token-by-token starting with a pre-filled KV cache')
    parser.add_argument('-m', '--model', type=str, default='decapoda-research/llama-7b-hf', help='Model name')
    parser.add_argument('-d', '--device', choices=['huggingface', 'pytorch', 'golden', 'silicon'], default='huggingface', help='huggingface: run using HF code only, pytorch: use our shim but run in PyTorch, golden/silicon: run via pybuda')
    parser.add_argument('--arch', choices=['greyskull', 'wormhole_b0'], default='wormhole_b0', help='Architecture to use for silicon')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16', 'fp8', 'fp8b'], default='fp32', help='Precision to use for all silicon tensors')
    parser.add_argument('--amp-level', type=int, default=0, choices=[0, 1, 2], help='Automatic mixed precision level (0=off, 1=mixed b-formats, 2=mixed a-formats)')
    parser.add_argument('--num-chips', type=int, default=1, help='Number of chips to use')
    parser.add_argument('--fuse', action='store_true', help='Fuse layers')
    parser.add_argument('--perf', choices=['none', 'light', 'verbose'], default=None, help='Performance tracing')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'TRACE'], default='ERROR', help='Log level')
    parser.add_argument('--context-length', type=int, default=2048, help='Context length')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers to use for the model')
    parser.add_argument('--opt-level', type=int, default=0, help='Optimization level for silicon')
    parser.add_argument('--loops', type=int, default=-1, help='If set, only read the first input and loop over that')
    parser.add_argument('--verify', action='store_true', help='If set, verify the results against the original model')

    args = parser.parse_args()

    # Load the model
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    model = AlpacaForCausalLM.from_pretrained(args.model, use_cache=True, return_dict=False, num_hidden_layers=args.num_layers)
    model.eval()

    seq_len = args.context_length
    num_windows = seq_len // 32

    input_ids = torch.arange(0, seq_len).unsqueeze(0)

    # For compilation only
    prefill_kvs = []
    for _ in range(model.config.num_hidden_layers):
        prefill_kvs.append(torch.zeros((1, 1, seq_len - 32, model.config.hidden_size)))
        prefill_kvs.append(torch.zeros((1, 1, seq_len - 32, model.config.hidden_size)))
    past_key_cache_size = prefill_kvs[0].size()

    print('PyBeautify decoder blocks')
    netlist_name = f'llama_{args.precision}_{args.num_chips}_{args.num_layers}_{args.context_length}'
    model.model.blocks = PyBudify(model.model.blocks, device=args.device, arch=args.arch, precision=args.precision, amp_config_file="amp_configs/w3.json",
        num_chips=args.num_chips, fuse=args.fuse, perf=args.perf, log_level=args.log_level, prefill_kvs=prefill_kvs, num_layers=args.num_layers,
        netlist_name=netlist_name, opt_level=args.opt_level, verify=args.verify)


    with torch.no_grad():

        for chunk in range(args.loops):
            print('Chunk', chunk)
            finished_tokens = 0

            attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
            attention_mask[0,-32:] = 1 # Always attend to the appended window of tokens
            model.model.blocks.write_index = 0
            
            for window in range(num_windows):

                input_ids_window = input_ids[:, finished_tokens:finished_tokens + 32]
                position_ids = torch.arange(finished_tokens, finished_tokens + 32).unsqueeze(0) # Needs batch dimension
                output = model(input_ids_window, attention_mask, position_ids, past_key_cache_size=past_key_cache_size)

                finished_tokens += 32
                model.model.blocks.write_index = (model.model.blocks.write_index + 1) % (num_windows - 1)
                attention_mask[:, :finished_tokens] = 1



if __name__ == '__main__':
    main()

