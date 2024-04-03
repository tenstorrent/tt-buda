# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
import torch
from modeling_alpaca_caching import AlpacaForCausalLM
from transformers import LlamaConfig, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import time

from pybudify_caching import PyBudify
import pybuda

from prettytable import PrettyTable


def main():

    parser = ArgumentParser('Generate text token-by-token starting with a pre-filled KV cache')
    parser.add_argument('-m', '--model', type=str, default='decapoda-research/llama-7b-hf', help='Model name')
    parser.add_argument('-k', '--kv-cache', type=str, default='kv_cache.pt', help='KV cache file')
    parser.add_argument('-s', '--stop', type=str, default='\n\n', help='Text to stop decoding after')
    parser.add_argument('-n', '--num-tokens', type=int, default=10, help='Maximum number of tokens to generate')
    parser.add_argument('--output-at-end', action='store_true', help='Output at the end of generation instead of token by token')

    parser.add_argument('-d', '--device', choices=['huggingface', 'pytorch', 'golden', 'silicon'], default='huggingface', help='huggingface: run using HF code only, pytorch: use our shim but run in PyTorch, golden/silicon: run via pybuda')
    parser.add_argument('--no-kv-cache', action='store_true', help='Do not use a kv-cache and only generate the first 32 tokens')
    parser.add_argument('--arch', choices=['greyskull', 'wormhole_b0'], default='wormhole_b0', help='Architecture to use for silicon')
    parser.add_argument('--num-chips', type=int, default=1, help='Number of chips to use')
    parser.add_argument('--fuse', action='store_true', help='Fuse layers')
    parser.add_argument('--perf', choices=['none', 'light', 'verbose'], default=None, help='Performance tracing')
    parser.add_argument('--verify', action='store_true', help='Verify results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'TRACE'], default='ERROR', help='Log level')
    parser.add_argument('--load', type=str, help='Load a TTImage')
    parser.add_argument('--save', type=str, help='Save a TTImage')
    parser.add_argument('--no-load-weights', action='store_true', help='Do not load weights from the model')
    parser.add_argument('--amp-config', type=str, help='AMP config file')
    parser.add_argument('--placement-config', type=str, help='Placement config file')
    parser.add_argument('--opt-level', type=int, default=0, help='Optimization level for silicon')

    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers to use')
    parser.add_argument('--context-length', type=int, default=2048, help='Context length')

    parser.add_argument('--validate', action='store_true', help='Validate on the fly with pytorch')
    parser.add_argument('--validate-kv', action='store_true', help='Validate KV-Cache on the fly with pytorch')

    #args for batching
    parser.add_argument('--rand-prompts', action='store_true', help='Generate args.batch_size random prompts of length args.seq_length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size or Number of Sequences. Used for debugging atm. To be removed later with real examples')
    parser.add_argument('--seq-length', type=int, default=5, help='Length of sequences in a batch. Used for deubgging atm. To be removed later with real examples. Only useful with args.rand_prompts')

    parser.add_argument('--prompt', type=str, default="Once upon a time, in a ", help='Prompt to start decoding with')

    parser.add_argument('--stop-on-eos', action='store_true', help='Stop generation on encountering end of sequence token.')

    args = parser.parse_args()

    print(f'args: {args}')
    # assert args.no_kv_cache, 'Only --no-kv-cache mode is supported for now'

    if args.device == 'huggingface':
        assert args.no_kv_cache, 'huggingface device only supports --no-kv-cache mode for now'

    # Download the model and tokenizer
    use_cache = not args.no_kv_cache
    gen_config = GenerationConfig.from_pretrained(args.model)
    if args.no_load_weights:
        config = LlamaConfig.from_pretrained(args.model, use_cache=use_cache, return_dict=False, num_hidden_layers=args.num_layers)
        model = AlpacaForCausalLM(config)
        # Unfortunately, IDK how to NOT load a tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model)
    else:
        model = AlpacaForCausalLM.from_pretrained(args.model, use_cache=use_cache, return_dict=False, num_hidden_layers=args.num_layers)
        # model.to(dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(args.model)

    if args.validate:
        model_hf = LlamaForCausalLM.from_pretrained(args.model, use_cache=True, return_dict=False, num_hidden_layers=args.num_layers)

    tokenizer.pad_token = tokenizer.eos_token   # eos_token=1 != pad_token=0 are different - https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/generation_config.json
    pad_token_id = tokenizer(tokenizer.pad_token, return_tensors='pt')['input_ids'].item()  # pad token not defined in tokenizer_config, pick up from generation_config.json instead
    pad_token_id = gen_config.pad_token_id

    # config:
    # seq_len = model.config.max_position_embeddings
    seq_len = args.context_length
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    model.eval()

    # past_key_values is a tuple of (key, value) tensors, one for each layer
    # pybudify expects a flat list of padded key, value tensors, one for each layer

    # Tokenize input here. Pad to WINDOW_SIZE
    # TODO: Remove when we get prefill.py
    prompt = args.prompt
    tokenized = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    input_ids = tokenized.input_ids

    assert input_ids.size(-1) < 32, "Input prompt too long. Max length is 32 tokens."

    # import pdb; pdb.set_trace()

    if args.rand_prompts:
        # hacking in here to override input ids with some rand ids for args.batch_size sequences
        # input_ids = torch.randint(low=0, high=32000-1, size=(args.batch_size, args.seq_length))
        input_ids = torch.randint(low=0, high=32000-1, size=(1, args.seq_length))

    input_ids = input_ids.repeat(args.batch_size, 1)        # same sequence passed in as two samples. easier debugging

    if args.validate:
        input_ids_hf = input_ids #torch.clone(input_ids)

    # frac_factor = 8

    # pybuda.config.insert_fracture_group([
    #     (f"matmul_{86}",-1, frac_factor),
    #     ]
    # )

    # for f in range(frac_factor):
    #     pybuda.config.configure_mixed_precision(
    #         output_df=pybuda.DataFormat.Bfp8_b,
    #         name_regex=f"fractured_{f}_matmul_86",
    #         input_df={0: [pybuda.DataFormat.Bfp8_b, True]},
    #         )

    # pybuda.config.configure_mixed_precision(
    #     output_df=pybuda.DataFormat.Bfp2_b,
    #     accumulate_df=pybuda.DataFormat.Bfp2_b,
    #     name_regex="fractured_gather_n0_matmul_86.dc.concatenate.0",
    #     intermediate_df=pybuda.DataFormat.Bfp2_b,
    #     input_df={0: [pybuda.DataFormat.Bfp2_b, True]},
    #     )

    # pybuda.config.configure_mixed_precision(
    #     output_df=pybuda.DataFormat.Bfp2_b,
    #     accumulate_df=pybuda.DataFormat.Bfp2_b,
    #     name_regex="llama_nonkv_1nc_1nl_128cl_1bsz.output_reshape_87",
    #     intermediate_df=pybuda.DataFormat.Bfp2_b,
    #     input_df={0: [pybuda.DataFormat.Bfp2_b, True]},
    #     )

    num_tokens = input_ids.shape[-1]
    print(f'input_ids :{input_ids.size()}')
    print(input_ids)    

    if not args.no_kv_cache:
        '''
        # Load the KV cache
        input_ids, past_key_values = torch.load(args.kv_cache)
        
        prefill_kvs = []
        for i, (k, v) in enumerate(past_key_values):
            # Pad the key and value tensors to seq_len tokens; they key and value are already in the same order (batch, heads, seq_len, head_dim)

            # TODO: we have a problem if we have >seq_len-32 tokens in the cache
            assert k.shape[1] <= seq_len - 32, "KV cache greater than seq_len-32 tokens is not implemented yet. LOL. D:"

            pad_k = torch.zeros((k.shape[0], k.shape[1], seq_len - 32, head_dim), dtype=torch.float32)
            pad_v = torch.zeros((v.shape[0], v.shape[1], seq_len - 32, head_dim), dtype=torch.float32)

            pad_k[:, :, :k.shape[-2], :] = k
            pad_v[:, :, :v.shape[-2], :] = v

            # merge heads for pybuda loopback
            def merge(tensor):
                num_attention_heads = tensor.shape[-3]
                attn_head_size = head_dim
                tensor = tensor.permute(0, 2, 1, 3).contiguous() # -> [bs, seq_len, num_attention_heads, attn_head_size]
                tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
                return tensor

            pad_k = merge(pad_k)
            pad_v = merge(pad_v)

            prefill_kvs.append(pad_k)
            prefill_kvs.append(pad_v)
        '''
        prefill_kvs = []
        for _ in range(model.config.num_hidden_layers):
            prefill_kvs.append(torch.zeros((args.batch_size, 1, seq_len - 32, model.config.hidden_size)))
            prefill_kvs.append(torch.zeros((args.batch_size, 1, seq_len - 32, model.config.hidden_size)))

    start_time = None

    all_text = ''
    with torch.no_grad():
        # Now transition to running in token-by-token mode for generation
        new_tokens = 0

        # There are two parts to our input. The "finished" part which is a multiple of 32 tokens
        # and the "current" part which is a tile of 32 tokens, only the first k of which are valid.
        # The attention mask is an oddity - it is 1 for the finished tokens and 0 for the rest,
        # *except* that the final 32 entries in the mask are used for the mask of the current part
        # We track these as follows:
        #  num_tokens: total number of tokens including finished and current
        #  finished_tokens: number of tokens in the finished part
        #  current_token: index of tokens in the current part (max. 31)
        #  position_ids: 32-element tensor of position ids for the finished_tokens + 32
        #  input_ids: 32-element tensor of tokens for the current part, filled with pad tokens
        #  write_index: current tile index being written to (each tile has 32 entries)
        finished_tokens = (num_tokens // 32) * 32
        current_token = num_tokens % 32 - 1
        write_index = num_tokens // 32

        attention_mask = torch.zeros((args.batch_size, seq_len), dtype=torch.long, device=input_ids.device)
#        attention_mask = torch.zeros((1, seq_len), dtype=torch.long, device=input_ids.device)

        if args.no_kv_cache:
            # no-cache mode only supports 32 tokens of output right now
            attention_mask = torch.zeros((args.batch_size, 32), dtype=torch.long, device=input_ids.device)
            prefill_kvs = None

        if args.device != 'huggingface':
            print('PyBeautify whole model')
            amp_config_name_safe = args.amp_config.split("/")[-1].split('.')[0]
            netlist_name = f'llama_{amp_config_name_safe}_{args.num_chips}nc_{args.num_layers}nl_{args.context_length}cl_{args.batch_size}bsz'
            model.model.blocks = PyBudify(model.model.blocks, device=args.device, arch=args.arch, amp_config_file=args.amp_config,
                num_chips=args.num_chips, fuse=args.fuse, perf=args.perf, log_level=args.log_level, prefill_kvs=prefill_kvs, num_layers=args.num_layers,
                netlist_name=netlist_name, opt_level=args.opt_level, verify=args.verify, placement_config_file=args.placement_config)

        attention_mask[:, :finished_tokens] = 1

        position_ids = torch.arange(finished_tokens, finished_tokens + 32).unsqueeze(0)     # Needs batch dimension
        position_ids = position_ids.repeat(args.batch_size, 1)                              # for batching - assuming samples of equal length atm
        
        prefill_ids = input_ids[:, -(current_token + 1):]
        pad_ids = torch.full((args.batch_size, 32 - (current_token + 1)), pad_token_id, dtype=torch.long, device=input_ids.device)
        print(f'input_ids: {input_ids.shape}')
        print(f'prefill_ids: {prefill_ids.shape}')
        print(f'pad_ids: {pad_ids.shape}')
        print(f'attention_mask: {attention_mask.shape}')
        print(f'position_ids: {position_ids.shape}')
        input_ids = torch.cat([prefill_ids, pad_ids], dim=1)

        full_output = None
        full_output = prefill_ids
        past_key_cache_size = prefill_kvs[0].size()

        print(f'Flushing stdout', flush=True)

        decode_time_history = []
        emb_time_history = []
        lm_time_history = []

        if args.validate:
            pcc_batch_history = []
            pcc_history = []
            pcc_k_history = []
            pcc_v_history = []
            em_history = []
            top_5_history = []

            past_kv_hf = None
        
        while True:
            
            # import pdb; pdb.set_trace()

            # Break out if we've generated enough tokens
            if num_tokens >= args.context_length:
                print('## Generated as many tokens as we can due to context length ##')
                break
            if args.stop and args.stop in all_text:
                break
            
            if args.num_tokens and new_tokens >= args.num_tokens:
                break
            
            # Generate the next token. We track position_id to get correct positional embeddings for the current 32-token tile
            attention_mask[:, -32 : -32 + current_token + 1] = 1
            if current_token == 31:
                attention_mask[:, -32 :] = 1
            
            # # Call model
            print('####### In Decode Loop #######')
            print(f'num_tokens: {num_tokens}, finished_tokens: {finished_tokens}, current_token: {current_token}, write_index: {model.model.blocks.write_index}')
            print(f'input_ids: {input_ids.shape}',)
            print(f'input_ids: {input_ids.tolist()}')
            if args.validate:
                print(f'input_ids_hf: {input_ids_hf.shape}',)
                print(f'input_ids_hf: {input_ids_hf.tolist()}')
                if past_kv_hf is not None:
                    print(f'past_kv_hf_len: {len(past_kv_hf[0][0][0][0])}')
            print(f'attention_mask: {attention_mask.shape}')
            print(f'attention_mask: {attention_mask[:, -32:].tolist()}')
            print(f'position_ids: {position_ids.shape}')
            print(f'position_ids: {position_ids.tolist()}')
            print(f'prefill_kvs: {prefill_kvs[0].shape}')
            # position_ids=None
            # attention_mask = None
            if position_ids is None:
                print('WARNING: POSITION_IDS is NONE')
            if attention_mask is None:
                print('WARNING: ATTENTION_MASK is NONE')

            ### Run the model ###
            start = time.time()
            output = model(input_ids, attention_mask, position_ids, past_key_cache_size=past_key_cache_size, current_token=current_token)
            decode_time = time.time() - start

            decode_time_history.append(decode_time)
            emb_time_history.append(output[2])
            lm_time_history.append(output[3])
            #####################
            logits = output[0]

            # token = logits[:, current_token, :].argmax(dim=-1)
            token = logits[:, [0], :].argmax(dim=-1)

            if args.validate:
                # output_hf = model_hf(input_ids, att_mask, position_ids)
                if past_kv_hf is None:      # first run, there's no kv cache
                    output_hf = model_hf(input_ids_hf, past_key_values=past_kv_hf)
                else:                       # now that we've got kv cache, only pass in last input_id
                    output_hf = model_hf(prev_token, past_key_values=past_kv_hf)

                logits_hf = output_hf[0]
                past_kv_hf = output_hf[1]

                token_hf = logits_hf[:, -1, :].argmax(dim=-1)

                input_ids_hf = torch.cat([input_ids_hf, token_hf[:, None]], dim=-1)

                # pcc = torch.corrcoef(torch.stack([logits[:, current_token, :].view(-1), logits_hf[:, -1, :].view(-1)]))[0,1]
                # import pdb; pdb.
                # pcc = torch.corrcoef(torch.stack([logits[:, [0], :].reshape(-1), logits_hf[:, -1, :].reshape(-1)]))[0][1]     # reshape needed when batching
                # pcc_batch_history.append(pcc)

                # get pcc of kv cache - single decoder atm
                # ugggllyyy, get rid of numbers and parameterise later
                # pb_k = output[1][0].transpose(1, 2).reshape(args.batch_size, -1, 4096)[:, :output_hf[1][0][0].shape[-2], :].reshape(-1)-1
                # pb_v = output[1][1].transpose(1, 2).reshape(args.batch_size, -1, 4096)[:, :output_hf[1][0][1].shape[-2], :].reshape(-1)-1

                # pick out relevant kv cache according to indices where we pay attention i.e. attention_mask == 1
                # pb_k = output[1][0].transpose(1, 2).reshape(args.batch_size, -1, 4096)[:, attention_mask[0].nonzero().T.squeeze(), :]
                # pb_v = output[1][0].transpose(1, 2).reshape(args.batch_size, -1, 4096)[:, attention_mask[0].nonzero().T.squeeze(), :]

                # pcc_k_cache = torch.corrcoef(torch.stack([pb_k, output_hf[1][0][0].transpose(1, 2).reshape(args.batch_size, -1, 4096).reshape(-1)]))[0][1]
                # pcc_v_cache = torch.corrcoef(torch.stack([pb_v, output_hf[1][0][1].transpose(1, 2).reshape(args.batch_size, -1, 4096).reshape(-1)]))[0][1]
                pcc_k_cache = 0
                pcc_v_cache = 0
                # pcc_k_cache = torch.corrcoef(torch.stack([output[1][0].squeeze()[:, :output_hf[1][0][0].shape[-2]%32, :].reshape(-1)-1, output_hf[1][0][0][:, :, (output_hf[1][0][0].shape[-2]//32)*32:, :].transpose(1, 2).reshape(2, -1, 4096).reshape(-1)]))[0][1]
                # pcc_v_cache = torch.corrcoef(torch.stack([output[1][1].squeeze()[:, :output_hf[1][0][1].shape[-2]%32, :].reshape(-1)-1, output_hf[1][0][1][:, :, (output_hf[1][0][0].shape[-2]//32)*32:, :].transpose(1, 2).reshape(2, -1, 4096).reshape(-1)]))[0][1]

                # import pdb; pdb.set_trace()

                for n in range(output[0].shape[0]): # for samples in batch
                    
                    pcc = torch.corrcoef(torch.stack([logits[n, [0], :].reshape(-1), logits_hf[n, -1, :].reshape(-1)]))[0][1]     # reshape needed when batching
                    print(f'pcc s{n}: {pcc}')
                    pcc_batch_history.append(pcc)

                    em_history.append((token[n] == token_hf[n]).to(dtype=torch.float32).mean().item())

                    if args.validate_kv:
                        
                        k_pb = output[1][0][n, ...][:, attention_mask[0].nonzero().T.squeeze(), :] - 1
                        v_pb = output[1][1][n, ...][:, attention_mask[0].nonzero().T.squeeze(), :] - 1

                        k_pt = output_hf[1][0][0][n, ...]
                        v_pt = output_hf[1][0][1][n, ...]

                        pcc_k_cache = torch.corrcoef( torch.stack( [k_pb.reshape(-1), k_pt.reshape(-1)] ) )[0][1]
                        print(f'pcc k-cache s{n}: {pcc_k_cache}')
                        pcc_v_cache = torch.corrcoef( torch.stack( [v_pb.reshape(-1), v_pt.reshape(-1)] ) )[0][1]
                        print(f'pcc v-cache s{n}: {pcc_v_cache}')

                        pcc_k_history.append(pcc_k_cache)
                        pcc_v_history.append(pcc_v_cache)

                print(f'token_pb: {token}')
                print(f'token_hf: {token_hf}')
                # in_top_5 = check_top5(next_token_logits_pb, next_tokens)

                # pcc_history.append(pcc)
                # em_history.append((token == token_hf).item())
                # em_history.append((token == token_hf).to(dtype=torch.float32).mean().item())
                # top_5_history.append(in_top_5.item())

            prev_token = token

            # Start timer after first token
            if start_time is None:
                start_time = time.time()

            # Update the new and total token counts
            new_tokens += 1
            num_tokens += 1
            current_token += 1
            
            if current_token == 32: #31:
                if args.no_kv_cache:
                    print("Running --no-kv-cache only supports 32 tokens right now")
                    sys.exit(1)
                finished_tokens += 32
                current_token = 0

                position_ids = torch.arange(finished_tokens, finished_tokens + 32).unsqueeze(0)
                position_ids = position_ids.repeat(args.batch_size, 1)                              # for batching - assuming samples of equal length atm

                input_ids = torch.full((args.batch_size, 32), pad_token_id, dtype=torch.long, device=input_ids.device)

                model.model.blocks.write_index += 1 # HAAAAAAAACK

                attention_mask[:, :finished_tokens] = 1
                attention_mask[:, -32:] = 0

            # import pdb; pdb.set_trace()

            # Update the input with the new token after checking current_token index
            input_ids[:, current_token] = token

            # Update full output for later decoding
            full_output = torch.cat([full_output, token], dim=1)

            # commenting out for batching
            # # Print the generated token
            # text = tokenizer.decode(token, skip_special_tokens=True)
            # if not args.output_at_end:
            #     print(text, end='', flush=True)
            # all_text += text

            pt = PrettyTable()
            pt.field_names = ["Metric", "Mean", "Std", "Min", "Max"]

            pt.add_row([f"num_tokens", num_tokens*args.batch_size, 'N/A', 'N/A', 'N/A'] )
            if len(decode_time_history) > 1:
                dec = torch.tensor(decode_time_history[1:])
                emb_time = torch.tensor(emb_time_history[1:])
                lm_time = torch.tensor(lm_time_history[1:])
                pt.add_row([f"Decode Time (s)", dec.mean().item(), dec.std().item(), dec.min().item(), dec.max().item()])
                pt.add_row([f"Embedding Time (s)", emb_time.mean().item(), emb_time.std().item(), emb_time.min().item(), emb_time.max().item()])
                pt.add_row([f"LM Head Time (s)", lm_time.mean().item(), lm_time.std().item(), lm_time.min().item(), lm_time.max().item()])

            if args.validate:

                # em = torch.tensor(em_history)
                pcc = torch.tensor(pcc_batch_history)
                # top_5 = torch.tensor(top_5_history)

                # pt.add_row([f"PCC", pcc.mean().item(), pcc.std().item(), pcc.min().item(), pcc.max().item()])
                for n in range(args.batch_size):
                    pcc = torch.tensor(pcc_batch_history[n::args.batch_size])
                    pt.add_row([f"PCC s{n}", pcc.mean().item(), pcc.std().item(), pcc.min().item(), pcc.max().item()])

                    em = torch.tensor(em_history[n::args.batch_size])
                    pt.add_row([f"EM", torch.mean(em.float()).item(), 'N/A', 'N/A', 'N/A'])

                    if args.validate_kv:
                        pcc_k = torch.tensor(pcc_k_history[n::args.batch_size])
                        pcc_v = torch.tensor(pcc_v_history[n::args.batch_size])
                        pt.add_row([f"PCC K-cache s{n}", pcc_k.mean().item(), pcc_k.std().item(), pcc_k.min().item(), pcc_k.max().item()])
                        pt.add_row([f"PCC V-cache s{n}", pcc_v.mean().item(), pcc_v.std().item(), pcc_v.min().item(), pcc_v.max().item()])

            print(pt, flush=True)

            for n in range(args.batch_size):
                print( f'Decode s{n}: {tokenizer.decode(full_output[n], skip_special_tokens=False)}' )

            if args.stop_on_eos:
                if token[0, None] == gen_config.eos_token_id:  # end of sequence token encountered
                    break

        # Get the total time taken
        total_time = time.time() - start_time

        print('===== decode.py summarizes final output =====')
        # print(tokenizer.decode(full_output.squeeze(), skip_special_tokens=False))
        for n in range(full_output.shape[0]):
            print(tokenizer.decode(full_output[n, :], skip_special_tokens=False))
        print(f'Generated {new_tokens} tokens in {total_time:.3f}s ({new_tokens / total_time:.2f} tokens/s), ms per token: {total_time / new_tokens * 1000:.3f}ms') 

if __name__ == '__main__':
    main()
