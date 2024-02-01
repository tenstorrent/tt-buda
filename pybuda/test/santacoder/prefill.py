# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#from pybudify import PyBudify
#from monkeypatch import monkeypatch


def main():
    parser = ArgumentParser('Prefill KV cache with a pretrained model')
    parser.add_argument('-m', '--model', type=str, default='bigcode/santacoder', help='Model name')
    parser.add_argument('-p', '--prompt', type=str, default='def hello_world():', help='Initial prompt for the model')
    parser.add_argument('-o', '--output', type=str, default='kv_cache.pt', help='Output file for the KV cache')
    args = parser.parse_args()

    if args.prompt == '-':
        args.prompt = input()

    # Download the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    with torch.no_grad():
        # Encode the prompt
        input_text = args.prompt
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Run on the prompt to get the first auto-generated token output and to fill the kv cache
        output = model(input_ids, use_cache=True, past_key_values=None)
        token = output.logits[:,-1].argmax(dim=-1)
        input_ids = token.unsqueeze(0)
        past_key_values = output.past_key_values 
        text = tokenizer.decode(token, clean_up_tokenization_spaces=True)
        print(input_text + text)

    torch.save((input_ids, past_key_values), args.output)
    print('Saved KV cache to', args.output)

if __name__ == '__main__':
    main()
