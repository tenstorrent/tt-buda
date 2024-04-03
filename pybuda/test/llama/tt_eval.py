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
from prettytable import PrettyTable

from pybudify_caching import PyBudify


def main():
    parser = ArgumentParser('Generate text token-by-token starting with a pre-filled KV cache')
    parser.add_argument('-m', '--model', type=str, default='decapoda-research/llama-7b-hf', help='Model name')
    parser.add_argument('-d', '--device', choices=['pytorch', 'golden', 'silicon'], default='huggingface', help='huggingface: run using HF code only, pytorch: use our shim but run in PyTorch, golden/silicon: run via pybuda')
    parser.add_argument('--arch', choices=['greyskull', 'wormhole_b0'], default='wormhole_b0', help='Architecture to use for silicon')
    parser.add_argument('--num-chips', type=int, default=1, help='Number of chips to use')
    parser.add_argument('--fuse', action='store_true', help='Fuse layers')
    parser.add_argument('--perf', choices=['none', 'light', 'verbose'], default=None, help='Performance tracing')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'TRACE'], default='ERROR', help='Log level')
    parser.add_argument('--context-length', type=int, default=2048, help='Context length')
    parser.add_argument('--input', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers to use for the model')
    parser.add_argument('--opt-level', type=int, default=0, help='Optimization level for silicon')
    parser.add_argument('--verify', action='store_true', help='If set, verify the results against the original model')
    parser.add_argument('--amp-config', type=str)
    parser.add_argument('--input-count', type=int)
    parser.add_argument('--nlp-target-cycles', type=int, default=-1)


    args = parser.parse_args()
    eval(vars(args))


def eval(args):
    # Load the model
    tokenizer = LlamaTokenizer.from_pretrained(args['model'], local_files_only=True, cache_dir="/proj_sw/large-model-cache/llama-offline")
    model = AlpacaForCausalLM.from_pretrained(args['model'], use_cache=True, return_dict=False, num_hidden_layers=args['num_layers'], local_files_only=True, cache_dir="/proj_sw/large-model-cache/llama-offline")
    model.eval()

    seq_len = args['context_length']
    window_size = 32
    num_windows = seq_len // window_size

    check_input_sequence_length(args['input'] + '/chunk_inputs_0.pt', seq_len)
    num_files = get_num_files(args['input'])
    print('Detected {} samples in input folder'.format(num_files))
    if args['input_count'] is not None:
        assert args['input_count'] <= num_files, 'Input count cannot be greater than number of files'
    num_files = num_files if args['input_count'] is None else args['input_count']

    start_time = None

    # For compilation only
    prefill_kvs = []
    cache_len = seq_len - window_size
    past_key_cache_size = (1, 1, cache_len, model.config.hidden_size)
    for _ in range(model.config.num_hidden_layers):
        prefill_kvs.append(torch.zeros(past_key_cache_size))
        prefill_kvs.append(torch.zeros(past_key_cache_size))

    print('PyBeautify decoder blocks')
    amp_config_name_safe = args['amp_config'].split("/")[-1].split('.')[0]
    netlist_name = f"llama_{amp_config_name_safe}_{args['num_chips']}_{args['num_layers']}_{args['context_length']}"
    model.model.blocks = PyBudify(model.model.blocks, device=args['device'], arch=args['arch'], amp_config_file=args['amp_config'],
        num_chips=args['num_chips'], fuse=args['fuse'], perf=args['perf'], log_level=args['log_level'], prefill_kvs=prefill_kvs, num_layers=args['num_layers'],
        netlist_name=netlist_name, opt_level=args['opt_level'], verify=args['verify'])

    stats = Stats()
    durations = []

    with torch.no_grad():
        # For each chunk of the evaluation dataset, process the input in chunks of 32
        # to get logits of the last token.

        for chunk in range(num_files):
            # load chunk_inputs_*.pt and chunk_logits_*.pt
            input_ids = torch.load(args['input'] + f'/chunk_inputs_{chunk}.pt')
            logits = torch.load(args['input'] + f'/chunk_logits_{chunk}.pt').float()

            # KV cache maintenance
            attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
            attention_mask[:,-window_size:] = 1 # Always attend to the appended window of tokens
            model.model.blocks.write_index = 0 # Reset write index

            for window in range(num_windows):
                finished_tokens = window * window_size
                # Before processing, update attention mask to attend to past cache
                attention_mask[:, :finished_tokens] = 1

                input_ids_window = input_ids[:, finished_tokens:finished_tokens + window_size]
                position_ids = torch.arange(finished_tokens, finished_tokens + window_size).unsqueeze(0) # Needs batch dimension

                start = time.time()
                output = model(input_ids_window, attention_mask, position_ids, past_key_cache_size=past_key_cache_size)
                end = time.time()
                durations.append(end - start)

                logits_window = output[0]
                observed_logits = logits_window[0, -1, :] # logits of last token

                expected_logits = logits[finished_tokens + 31, :]

                # Since cache size is (seqlen - 32), ensure that we don't go out of bounds. Could cause a nasty error.
                model.model.blocks.write_index = (model.model.blocks.write_index + 1) % (num_windows - 1)


            stats.collect_stats(expected_logits, observed_logits)
            stats.print_stats(chunk)
            if len(durations) > 1:
                print(f'Computed {len(durations) - 1} in {sum(durations[1:])} seconds')
                print(f'Window throughput: {(len(durations)-1) / sum(durations[1:])} windows per second')

    mean_pcc = torch.tensor(stats.pcc).float().mean().item()
    print('Mean PCC:', mean_pcc)
    assert mean_pcc >= 0.41604292392730713  # TODO


def check_input_sequence_length(input_ids_path, seq_len):
    assert torch.load(input_ids_path).shape[1] == seq_len, "Input sequence length does not match model context length"

def get_num_files(input_dir):
    # Find number of chunk_logits_*.pt files
    num_files = 0
    while True:
        try:
            torch.load(input_dir + f'/chunk_logits_{num_files}.pt')
            num_files += 1
        except FileNotFoundError:
            break

    return num_files

class Stats:
    def __init__(self):
        self.pcc = []
        self.top_1 = []
        self.top_5 = []
        self.kl_divergence = []

    def collect_stats(self, expected, observed):
        assert expected.shape == observed.shape, "Expected and observed logits must have the same shape"
        assert expected.dim() == 1, "Expected and observed logits must be 1D tensors. Received tensors of shape {}".format(expected.shape)

        # Compute PCC
        pcc = torch.corrcoef(torch.stack([expected.view(-1), observed.view(-1)]))[0,1].item()
        self.pcc.append(pcc)

        # Compute top-1 accuracy
        top_1 = (expected.argmax(dim=-1) == observed.argmax(dim=-1)).float().item()
        self.top_1.append(top_1)

        # Compute top-5 accuracy
        top_5 = torch.topk(expected, 5, dim=-1).indices.view(-1)
        top_5 = (top_5 == observed.argmax(dim=-1)).any(dim=-1).float().item()
        self.top_5.append(top_5)

        # Compute KL divergence
        kl_expected = torch.nn.functional.log_softmax(expected, dim=-1)
        kl_observed = torch.nn.functional.log_softmax(observed, dim=-1)
        kl_divergence = torch.nn.functional.kl_div(kl_observed.unsqueeze(0), kl_expected.unsqueeze(0), log_target=True).item()
        self.kl_divergence.append(kl_divergence)

    def print_stats(self, chunk):
        pt = PrettyTable()
        print(f'Up to chunk {chunk} statistics:')
        pt.field_names = ["Metric", "Mean", "Std"]
        pt.add_row(["PCC",] + self.mean_std_from_list(self.pcc))
        pt.add_row(["KL Divergence",] + self.mean_std_from_list(self.kl_divergence))
        pt.add_row(["Top 1",] + self.mean_std_from_list(self.top_1))
        pt.add_row(["Top 5",] + self.mean_std_from_list(self.top_5))
        print(pt, flush=True)

    def mean_std_from_list(self, stat):
        stat = torch.tensor(stat).float()
        return [stat.mean().item(), stat.std().item()]


if __name__ == '__main__':
    main()

