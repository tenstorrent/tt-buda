# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
'''
Generate PyTorch ground truth logits for the eval dataset.
'''
from transformers import LlamaConfig, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import argparse
import os
import torch


def generate_eval(args):
    assert os.path.isfile(args['input'])

    # Make output dir if it doesn't already exist. If it does, check that it's empty.
    if not os.path.isdir(args['output']):
        print(f"Creating output dir {args['output']}")
        os.makedirs(args['output'])
    else:
        if len(os.listdir(args['output'])) > 0:
            print(f"Output dir {args['output']} is not empty, skipping generate_eval")
            return

    # Load the model
    print(f"Loading model from {args['model']}")
    print(f"Running with layers: {args['num_layers']}")
    tokenizer = LlamaTokenizer.from_pretrained(args['model'], local_files_only=True, cache_dir="/proj_sw/large-model-cache/llama-offline")
    model = LlamaForCausalLM.from_pretrained(args['model'], num_hidden_layers=args['num_layers'], local_files_only=True, cache_dir="/proj_sw/large-model-cache/llama-offline")
    model.eval()

    if args['device'] == 'cuda':
        model.half() # half is only supported on cuda machines

    model.to(args['device'])


    # Load the input text file and tokenize it fully
    with open(args['input'], 'r') as f:
        text = f.read()
    tokenized_text = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    input_ids = tokenized_text.input_ids
    print(f'input_ids: {input_ids}')

    # Randomly pick num-samples starting indices and generate that many samples of context-length
    print(f"Generating {args['num_samples']} samples")
    indices = torch.randint(0, input_ids.size(1)-args['context_length'], (args['num_samples'],))

    chunks = []
    for i in range(args['num_samples']):
        chunk = input_ids[:, indices[i]:indices[i]+args['context_length']]
        chunks.append(chunk)
        print(f'chunk: {input_ids.size()}')
        # Save each chunk to file
        torch.save(chunk, os.path.join(args['output'], f'chunk_inputs_{i}.pt'))

    chunks = torch.cat(chunks, dim=0)

    print(f'chunks: {chunks.size()}')
    with torch.no_grad():
        # Generate logits for each chunk
        for i, chunk in enumerate(chunks):
            chunk = chunk.to(args['device'])
            print(f'Generating logits for chunk {i}')
            logits = model(chunk.unsqueeze(0)).logits.squeeze(0)
            logits = logits.cpu()
            print(f'logits: {logits.size()}')
            # Save logits to file
            torch.save(logits, os.path.join(args['output'], f'chunk_logits_{i}.pt'))
            del(logits)

    print('Done generate_eval.')


def main():
    # argparse stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='decapoda-research/llama-7b-hf', help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output dir')
    parser.add_argument('--context-length', type=int, default=128, help='Context length')
    parser.add_argument('--num-samples', type=int, help='Number of samples to generate')
    parser.add_argument('--device', default='cpu', help='Device to run on')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers in the model')

    # Check that input file exists
    args = parser.parse_args()
    generate_eval(vars(args))


if __name__ == '__main__':
    main()
