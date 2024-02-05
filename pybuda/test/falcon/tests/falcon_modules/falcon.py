# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import sys
import os
from time import time
from datetime import datetime
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import json
from peft import PeftModel
import torch
from torch.utils.data import DataLoader

from ..utils import get_tokenizer, lorify_model, get_falcon_model_with_version, set_random_seed, get_mmlu_eval_dataloader, create_experiment_dir, make_data_module, get_gradient_norm, get_lora_short_name_from_modules


def run_sync(model, tokenizer, input_ids, past_key_values, num_tokens, seqlen, n_layers):
    dest = 'cpu'
    model.to(dest)
    all_text = ''
    new_tokens = 0

    batch_size = 1

    output_ids = input_ids.clone()
    output_hidden_states = None

    while True:
        if new_tokens >= 100:
            break

        # Generate the next token. We track position_id to get correct positional embeddings.
        position_ids = torch.tensor([[num_tokens]], dtype=torch.long, device=input_ids.device)
        attention_mask = torch.ones((1, seqlen), dtype=torch.long, device=input_ids.device)
        attention_mask[:, num_tokens:-1] = 0 # Mask out unused tokens; current token is appended to cache in attention computation

        # single user
        # expand the batch dimension to match the desired batch size
        input_ids = input_ids.expand(batch_size, -1).to(dest)
        position_ids = position_ids.expand(batch_size, -1).to(dest)
        attention_mask = attention_mask.expand(batch_size, -1).to(dest)
        new_past_key_values = []
        for (k,v) in past_key_values:
            k = k.expand(batch_size, k.size(1), k.size(2)).clone().to(dest)
            v = v.expand(batch_size, v.size(1), v.size(2)).clone().to(dest)
            new_past_key_values.append([k,v])
        past_key_values = tuple(new_past_key_values)

        outputs = model(input_ids, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        output = outputs[0]
        new_key_values = outputs[1] # tuple of (k_0, v_0, k_1, v_1, ...)
        assert len(new_key_values) == 2 * n_layers, f"new_key_values has wrong number of layers, expected {2 * n_layers}, got {len(new_key_values)}"
        output = output.to('cpu')

        token = output[-1].argmax(dim=-1)
        input_ids = token.unsqueeze(0)
        output_ids = torch.cat((output_ids, token.unsqueeze(0)), dim=1)

        if output_hidden_states is None:
            output_hidden_states = output
        else:
            output_hidden_states = torch.cat((output_hidden_states, output), dim=0)

        # Insert the new key, value into the past_key, past_value tensors at the right position
        for i, (k,v) in enumerate(past_key_values):
            k_new = new_key_values[i*2]
            v_new = new_key_values[i*2+1]
            k[:,[num_tokens],:] = k_new
            v[:,[num_tokens],:] = v_new

        # Update the new and total token counts
        new_tokens += 1
        num_tokens += 1

    # save hidden states outputs
    # torch.save(output_hidden_states, f'silicon_{n_layers}L_hidden_states.pt')

    all_text = tokenizer.decode(output_ids[0], clean_up_tokenization_spaces=True)

    return all_text


def run_demo_sync_finetune(model, tokenizer, dataloader, num_layers, sequence_length):
    dest = 'cpu'
    model.to(dest)

    # Parameters
    user_rows = 32
    max_tokens = 256
    batch_size = 1
    seqlen = sequence_length
    layers = num_layers
    temperature = 1.0
    
    batch_idx = 0 # dataset batch index
    generator_name = f"pybuda-greedy-alpaca_eval-ci"

    sequence_nlls = [] # list of negative log likelihood

    # samples is a batch of 32 dataset entries (represented by 32 users)
    for samples in dataloader:
        user_nlls = [[] for i in range(user_rows)]

        prompts = samples['input']
        instructions = samples['instruction']
        ground_truth = samples['output']
        generators = [generator_name for i in range(user_rows)]
        dsets = samples['dataset']
        
        batch_length = len(instructions)

        for i in range(user_rows):
            if i >= batch_length: # Account for special case where batched prompts < user_rows
                # We always process 32 users. If there's no prompts to fill them, we fill it instead with "empty" and avoid processing them
                prompts.append("empty")
                instructions.append("empty")
                ground_truth.append("")
                generators.append("ignore")
                dsets.append("None")

        # Tokenize the prompts and attention mask
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        tokenized_input_ids = tokenized.input_ids.unsqueeze(0).clone()
        tokenized_attention_mask = tokenized.attention_mask.unsqueeze(0).clone()
        prompt_token_counts = [sum(mask) for mask in tokenized.attention_mask]

        # initial attention mask, rows will be overwritten for prefill users
        attention_mask = torch.zeros((batch_size, user_rows, seqlen), dtype=torch.long, device=tokenized_attention_mask.device).to(dest)
        # Set initial input_ids to first token
        input_ids = tokenized_input_ids[:, :, 0].clone()

        # When calculating perplexity, also tokenize the ground truth from the dataset
        # Account for special case where batched prompts < user_rows
        # In this case, set the ground truth to EOS token
        for i in range(user_rows):
            if i >= batch_length:
                ground_truth.append(tokenizer.eos_token)

        calc_ppl = False # Flag that controls when we start considering the loss calculations (i.e. after at least 1 user is generating outputs)
        tokenized_ground_truth = tokenizer(ground_truth, padding=True, return_tensors="pt")
        tokenized_ground_truth_input_ids = tokenized_ground_truth.input_ids.unsqueeze(0).clone()
        target_ids = input_ids.clone() # initial target will be the same as input, since we start in prefill mode

        assert min(prompt_token_counts) > 0, "Empty prompts for unconditional generation not currently supported"
        assert batch_size == 1

        # tensor of right size and shape needed for pybuda to compile. initialise kv with zeros
        # value in tensor doesn't matter. we're going to prefill this in anyways
        past_key_values = tuple([(torch.zeros((batch_size, 1, seqlen, 64 * user_rows)), # [batch, 1, seqlen, head_sim * num_users]
                                    torch.zeros((batch_size, 1, seqlen, 64 * user_rows))) # [batch, 1, seqlen, head_sim * num_users]
                                    for _ in range(layers)])


        current_token = 0 # index of the token being prefilled/generated
        end_token_pos = [ None for _ in range(user_rows) ]
        all_users_done = 0 # Stop condition flag to keep track when all users are done generating outputs

        # When a batch has fewer prompts than users, set the idle users to done
        if batch_length < user_rows:
            for i in range(user_rows):
                if i >= batch_length:
                    end_token_pos[i] = 0

        print(f'\n >> Processing batch {batch_idx} of {len(dataloader)-1}')
        # Keep generating until all users finished their output or until the number of max tokens is reached (not counting initial prompt)
        while (not all_users_done) and (current_token < max_tokens + max(prompt_token_counts)):
            # Override any attention and input rows for users that are still prefilling from their prompt
            for i in range(user_rows):
                if current_token < prompt_token_counts[i]:
                    attention_mask[:, i, current_token%seqlen] = tokenized_attention_mask[:, i, current_token]
                    # prefill mode picks input_ids from the prompt tokens
                    input_ids[:, i] = tokenized_input_ids[:, i, current_token]
                else:   # prefill has finished for user. Set attention mask to 1 for next token while rolling over
                    attention_mask[:, i, current_token%seqlen] = 1

                # Perplexity
                if current_token < prompt_token_counts[i]-1:
                    target_ids[:,i] = -100 # Ignore the prefill labels
                else:
                    # Labels will match the ground truth after user finishes prefill and starts generation
                    if current_token-prompt_token_counts[i] + 1 < tokenized_ground_truth_input_ids.shape[2]:
                        # if ground truth output reaches end of text, ignore loss for that user
                        if tokenized_ground_truth_input_ids[:, i, current_token-prompt_token_counts[i]+1] == tokenizer.eos_token_id:
                            target_ids[:,i] = -100
                        else:
                            target_ids[:,i] = tokenized_ground_truth_input_ids[:, i, current_token-prompt_token_counts[i]+1]
                    else: # If current token is larger than length of ground truth, ignore loss for that user
                        target_ids[:,i] = -100

            # As we use right-padding all users have the same current_token
            # We make sure that the position ids never go over seqlen
            position_ids = torch.tensor([[current_token%seqlen]], dtype=torch.long, device=tokenized_input_ids.device)
            position_ids = position_ids.expand(batch_size, user_rows).clone().to(dest)

            kv_read_mask = torch.ones((1, 1, seqlen, 1), device=input_ids.device)
            kv_read_mask[:, :, [current_token%seqlen], :] = 0

            kv_write_mask = torch.zeros((1, 1, seqlen, 1), device=input_ids.device)
            kv_write_mask[:, :, [current_token%seqlen], :] = 1

            outputs = model(input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            labels=target_ids,
                            kv_read_mask=kv_read_mask,
                            kv_write_mask=kv_write_mask
                            )


            logits = outputs.logits.to('cpu') # FIXME: doesn't this assume args.batch_size == 1?
            logits /= temperature

            # Only calculate loss if there's at least 1 user already generating output
            for i in range(user_rows):
                if current_token >= prompt_token_counts[i] - 1:
                    calc_ppl = True
            if calc_ppl:
                neg_log_likelihood = outputs.loss.to('cpu').float()
                for i in range(user_rows):
                    if target_ids[:, i] != -100:
                        user_nlls[i].append(neg_log_likelihood[i])

            # Greedy output
            token = logits.argmax(dim=-1)

            # Update the end of generation when EOS or BOS is detected. For the finetune demo we override BOS to '###'
            for i in range(user_rows):
                # Only stop generation after prefill is completed
                if current_token >= prompt_token_counts[i]-1:
                    if end_token_pos[i] is None:
                        # Stop conditions: Reach max num tokens (prompt + max_tokens) or reach EOS or reach '###' token
                        if (current_token >= (max_tokens + prompt_token_counts[i] - 1)) or (token[i] == tokenizer.eos_token_id) or (token[i] == 19468): # id=19468 == ###
                            end_token_pos[i] = current_token

            # Check if all users have reached the end of their generated output and stop
            if all(tok is not None for tok in end_token_pos):
                all_users_done=1

            # Update the current token index
            current_token += 1

            # Perplexity
            # If user is generating, set the new input_ids to the ground truth to keep calculating perplexity
            for i in range(user_rows):
                if current_token >= prompt_token_counts[i]:
                    if target_ids[:,i] == -100: # already reached end of ground truth
                        input_ids[:,i] = tokenizer.eos_token_id
                    else:
                        input_ids[:,i] = target_ids[:,i]


        # Finalize the current batch and cleanup things for next batch
        #perplexity
        batch_nlls = []
        for i in range(user_rows):
            if len(user_nlls[i])>0:
                sequence_loss = torch.stack(user_nlls[i]).mean()
                sequence_nlls.append(sequence_loss)
                batch_nlls.append(sequence_loss)
        avg_batch_nlls = torch.stack(batch_nlls).mean()
        avg_batch_ppl = torch.exp(avg_batch_nlls)
        print(f"avg batch {batch_idx} loss: {avg_batch_nlls}, ppl: {avg_batch_ppl}")

        avg_all_sequences_nlls = torch.stack(sequence_nlls).mean()
        avg_all_sequences_ppl = torch.exp(avg_all_sequences_nlls)
        print(f"avg dataset loss: {avg_all_sequences_nlls}, ppl: {avg_all_sequences_ppl}")

        # with open("PPL_Output.txt", "w", encoding='utf-8') as text_file:
        #     print(f"Loss: {avg_all_sequences_nlls}", file=text_file)
        #     print(f"Perplexity: {avg_all_sequences_ppl}", file=text_file)

        batch_idx += 1

    # Finetune decode demo CI evaluates the final perplexity scores.
    # TODO: The PPL baseline is currently hardcoded based on previously obtained scores
    # Check different of avg_all_sequences_ppl
    # if args.adapter:
        # ppl_baseline = 6.09985399246215
    ppl_baseline = 5.53517

    # Hardcoded range for good PPL
    ppl_range = 0.2
    assert abs(avg_all_sequences_ppl - ppl_baseline) < ppl_range, f"Average dataset PPL={avg_all_sequences_ppl} is not within a {ppl_range} score range of the expected baseline={ppl_baseline}"


def run_demo_sync_masked_odkv(model, tokenizer, tokenized, num_layers, sequence_length):
    dest = 'cpu'
    model.to(dest)

    # Parameters
    user_rows = 32
    max_tokens = 120
    batch_size = 1
    seqlen = sequence_length
    layers = num_layers
    temperature = 1.0

    tokenized_input_ids = tokenized.input_ids.unsqueeze(0).clone()
    tokenized_attention_mask = tokenized.attention_mask.unsqueeze(0).clone()
    prompt_token_counts = [sum(mask) for mask in tokenized.attention_mask]

    # initial attention mask, rows will be overwritten for prefill users
    # attention_mask = torch.ones((args.batch_size, args.user_rows, args.seqlen), dtype=torch.long, device=tokenized_attention_mask.device).to(dest)
    # initial attention mask, rows will be overwritten for prefill users
    attention_mask = torch.zeros((batch_size, user_rows, seqlen), dtype=torch.long, device=tokenized_attention_mask.device).to(dest)
    input_ids = tokenized_input_ids[:, :, 0].clone()

    assert min(prompt_token_counts) > 0, "Empty prompts for unconditional generation not currently supported"
    assert batch_size == 1, "Pretty sure this code assumes batch size == 1, FIXME"

    # tensor of right size and shape needed for pybuda to compile. initialise kv with zeros
    # value in tensor doesn't matter. we're going to prefill this in anyways
    # TODO: replace constants 32 and 64
    past_key_values = tuple([(torch.zeros((batch_size, 1, seqlen, 64 * user_rows)),
                                torch.zeros((batch_size, 1, seqlen, 64 * user_rows)))
                                for _ in range(layers)])

    num_tokens = 0
    all_output = [ tokenizer.decode(input_ids[0, i], clean_up_tokenization_spaces=True, skip_special_tokens=True).replace('\n', ' ')
                    for i in range(user_rows) ]
    end_token_pos = [ None for _ in range(user_rows) ]

    while True:
        if num_tokens >= max_tokens:
            break

        # Prepare attention_mask and position_ids for decode mode. We will override these for users still in prefill mode.
        # in decode phase we pay attention to all new tokens so concat 1 to attention_mask for the latest token
        # and shift out oldest tokens attention mask FIFO style similar to odkv cache logic
        # attention_mask = torch.cat((attention_mask, torch.ones((1, args.user_rows, 1))), dim=2)
        # attention_mask = attention_mask[:, :, 1:]

        # Now override any attention and input rows for users that are still prefilling from their prompt
        for i in range(user_rows):
            # in in prefill phase for any user, pick attention_mask from tokenized_attention_mask
            if num_tokens < prompt_token_counts[i]:
                # at the very least we have 1 prefill token so seqlen - 1 are the unused tokens. and then subtract num_tokens as we prefill them
                # mask out tokens which haven't been prefilled
                # attention_mask[:, i, :args.seqlen - num_tokens - 1] = 0

                attention_mask[:, i, num_tokens] = tokenized_attention_mask[:, i, num_tokens]

                # we get and set the prefill tokens attention mask according to the tokeniser which will mask out padded tokens for us
                # attention_mask[:, i, args.seqlen - num_tokens - 1:] = tokenized_attention_mask[:, i, :num_tokens + 1]
                # attention_mask[:, i, :num_tokens+1] = tokenized_attention_mask[:, i, :num_tokens+1]

                # prefill mode picks input_ids from the prompt tokens
                input_ids[:, i] = tokenized_input_ids[:, i, num_tokens]

            else:   # prefill is done for this user so just set attention mask to 1 for next token while rolling over
                attention_mask[:, i, num_tokens%seqlen] = 1

        # As we use right-padding all users have the same num_tokens
        position_ids = torch.tensor([[num_tokens]], dtype=torch.long, device=tokenized_input_ids.device)
        position_ids = position_ids.expand(batch_size, user_rows).clone().to(dest)

        kv_read_mask = torch.ones((1, 1, seqlen, 1), device=input_ids.device)
        kv_read_mask[:, :, [num_tokens%seqlen], :] = 0

        kv_write_mask = torch.zeros((1, 1, seqlen, 1), device=input_ids.device)
        kv_write_mask[:, :, [num_tokens%seqlen], :] = 1

        # outputs = model(input_ids, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        outputs = model(input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    kv_read_mask=kv_read_mask,
                    kv_write_mask=kv_write_mask
                    )
        logits = outputs[0].to('cpu') # FIXME: doesn't this assume args.batch_size == 1?
        logits /= temperature

        token = logits.argmax(dim=-1)
        
        # Use the expected output from the prompt as long as we are still prefilling; only the last token is from the model
        for i in range(user_rows):
            if num_tokens < prompt_token_counts[i] - 1:
                token[i] = tokenized_input_ids[0, i, num_tokens + 1]

        for i in range(user_rows):
            if end_token_pos[i] is None and token[i] == tokenizer.eos_token_id:
                end_token_pos[i] = num_tokens

        for i in range(user_rows):
            if end_token_pos[i] is None:
                all_output[i] += tokenizer.decode(token[i], clean_up_tokenization_spaces=True, skip_special_tokens=True)

        # if we're at the end of prefilling use the newly-generated token as input (overridden above if we are prefilling)
        input_ids = token.unsqueeze(0).to(dest) # FIXME: doesn't this assume args.batch_size == 1?

        # Update the new and total token counts
        num_tokens += 1

    for i in range(user_rows):
        print(f'User {i+1:02d}: {all_output[i]}')
    
    return all_output


wandb = None
host_device = "cuda:0" if torch.cuda.is_available() else 'cpu'

default_args = Namespace(
    learning_rate=2e-4,
    max_grad_norm=0.3,
    data_seed=42,
    model_seed=42,
    perf='none',
    sequence_length=128,
    pybuda_device='silicon',
    num_epochs=4,
    max_num_steps=0,
    num_layers=32,
    num_lora_layers=32,
    rank=64,
    precision='very-low-mp',
    version='padded_split',
    model='tiiuae/falcon-7b',
    lora_modules=["wq", "wv"],
    num_chips=1,
    batch_size=4,
    num_accumulation_steps=1,
    loss_scale=1.0,
    num_samples=0, # use the entire dataset
    dataset_name='guanaco_en_sp_fr',
    checkpoint_every_steps=None,
    checkpoint_at_steps=None,
    languages='en,es,fr',
    prefiltered_dataset_dir='/proj_sw/large-model-cache/falcon7b/datasets',
    activation_cache_file='/proj_sw/large-model-cache/falcon7b/cached_activations/cached_activations_bfp8b_quantW_fp16b_compute_pybuda.pth',
    num_cache_layers=16,
    wandb_log_steps=20,
    wandb_project='falcon-tuning',
    pybuda_log_level='ERROR',
    loss_on_device=False,
    optimizer_on_host=False,
    save_checkpoints_at_epoch_end=False,
    save_checkpoint_as_model_dict=False,
    save_optimizer_state=False
    )

# This is a copy of run_finetune.py
def run_finetune(cfg_file):
    global default_args
    global wandb

    parser = ArgumentParser('Fine-tune Falcon-7B using LoRA adapters.')

    # General
    parser.add_argument('--config', type=str, help='Path of a pre-defined training configuration')
    parser.add_argument('--with-pytorch', action='store_true', help='Train a model using pytorch')
    parser.add_argument('--with-reference', action='store_true', help='Compare trained models to saved checkpoint steps in the format golden_step_*.pt')
    parser.add_argument('--with-pybuda', action='store_true', help='Train a model using pybuda (use --device to select golden or silicon)')
    parser.add_argument('--pybuda-device', choices=['golden', 'silicon'], help='PyBuda argument only, run on golden/silicon')
    parser.add_argument('--num-chips', type=int, help='Number of chips to run on')

    # Training configurations
    parser.add_argument('--num-epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--max-num-steps', type=int, help='Maximum number of training steps (does not affect learning rate schedules or warmup, just stops training early)')
    parser.add_argument('--ignore-pad-token-loss', action='store_true', help='Whether to ignore padding tokens in the loss')
    parser.add_argument('--explicit-pad-token', action='store_true', help='Add explicit pad token')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--max-grad-norm', type=float, help='Cut off threshold for the gradient norm.')

    # Model configuration
    parser.add_argument('--model', type=str, help='Huggingface pre-trained model name')
    parser.add_argument('--version', type=str, choices=['huggingface', 'torch2.0', 'torch1.0', 'padded', 'fractured', 'padded_split', 'padded_split_tiny'], help='Specific Falcon-7B model version to use')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16','very-low-mp', 'almost-low-mp', 'low-mp', 'high-mp', 'debug'], help='Precision configuration')
    parser.add_argument('--num-layers', type=int, help='Number of layers in the model')
    parser.add_argument('--num-lora-layers', type=int, help='Last N layers to apply LoRA adapters to')
    parser.add_argument('--lora-modules', nargs='+', help='Modules to apply LoRA adapters to')
    parser.add_argument('--rank', type=int, help='Lora rank')
    parser.add_argument('--sequence-length', type=int, help='Pad to a fixed sequence length')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-accumulation-steps', type=int, help='Number of accumulation steps')
    parser.add_argument('--loss-scale', type=float, help='Scale loss by this amount')
    parser.add_argument('--model-seed', type=int, help='Seed used for model init')
    parser.add_argument('--mask-lora', action='store_true', help='Whether to mask padded activations after adding lora to account for zero padding')
    parser.add_argument('--load-in-8bit', action='store_true', help='Load model in 8bit')

    # Data
    parser.add_argument('--num-samples', type=int, help='Number of samples to load from the training set. Use "0" for the entire training set.')
    parser.add_argument('--data-seed', type=int, help='Seed used for data loader')
    parser.add_argument('--dataset-name', type=str, help='Dataset name')
    parser.add_argument('--filter-long-seq', action='store_true', help='Filter longer sequences')
    parser.add_argument('--languages', type=str, help='Languages to filter from datasets, empty -> no language filtering')
    parser.add_argument('--load-dataset-from-disk', action='store_true', help='Load prefiltered dataset from disk; uses --prefiltered-dataset-dir to load from')
    parser.add_argument('--prefiltered-dataset-dir', type=str, help='Path used to save/load the prefiltered dataset')
    parser.add_argument('--save-dataset-to-disk', action='store_true', help='Save prefiltered dataset to disk, use --prefiltered-dataset-dir to save to')

    # Data: caching activations
    parser.add_argument('--load-from-activation-cache', action='store_true', help='Whether to train from saved hidden states cache')
    parser.add_argument('--activation-cache-file', type=str, help='Cached hidden states filename')
    parser.add_argument('--save-to-activation-cache', action='store_true', help='Whether to save hidden states after --num-cache-layers to a file')
    parser.add_argument('--activation-cache-out-file', type=str, help='Hidden states cache output filename')
    parser.add_argument('--num-cache-layers', type=int, help='Number of decoders to cache in the model')
    parser.add_argument('--hf-cache', type=str, default='/proj_sw/user_dev/hf_data', help='Cache directory for huggingface')

    # Logging, checkpointing
    parser.add_argument('--verify', action='store_true', help='Enable pybuda verification code to compare tensors of intermediate pybuda passes')
    parser.add_argument('--save-final', action='store_true', help='Save final state_dicts for each trainer in "<trainer_name>.pt"')
    parser.add_argument('--checkpoint-at-steps', nargs='*', type=int, help='Save state_dicts for each trainer in "<trainer_name>_checkpoint_*.pt" for each step index contained in this comma-separated list. E.g. "100,200" for checkoints at step 100 and 200')
    parser.add_argument('--checkpoint-every-steps', type=int, help='Checkpoint every X steps')
    parser.add_argument('--dump-steps', action='store_true', help='Filename prefix to save state, gradient and optimizer dumps to for each step')
    parser.add_argument('--no-wandb', action='store_true', help='Do NOT log on WandB')
    parser.add_argument('--log-histograms', action='store_true', help='Include logging histograms of weights and gradients on WandB')
    parser.add_argument('--log-opt', action='store_true', help='Include logging optimizer statistics on WandB')
    parser.add_argument('--wandb-log-steps', type=int, help='Number of steps between logging on WandB')
    parser.add_argument('--wandb-log-name', type=str, help='WandB log name')
    parser.add_argument('--wandb-project', type=str, help='WandB log project name')
    parser.add_argument('--experiment-name', type=str, help='Unique name used to store log files and checkpoints to disk')
    parser.add_argument('--model-load-path', type=str, help='Load lora adapters from this directory')
    parser.add_argument('--optimizer-load-state-dict', type=str, help='Load optimizer from this state dict file')
    parser.add_argument('--save-checkpoints-at-epoch-end', action='store_true', help='Save checkpoint at the end of each epoch')
    parser.add_argument('--save-checkpoint-as-model-dict', action='store_true', help='Save checkpoint as model dict')
    parser.add_argument('--save-optimizer-state', action='store_true', help='Save optimizer state at checkpoints')
    parser.add_argument('--tti-load', type=str, help='Path to load a TTImage from')
    parser.add_argument('--tti-save', type=str, help='Path to save a TTImage to')


    # PyBuda and TTDevice specific options
    parser.add_argument('--optimizer-on-host', action='store_true', help='Run optimizer on host')
    parser.add_argument('--loss-on-device', action='store_true', help='Run loss on device')
    parser.add_argument('--pybuda-log-level', choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='PyBuda log level')
    parser.add_argument('--tt-build-dir', type=str, help='Use this custom TT build directory')
    parser.add_argument('--perf', choices=['none', 'light', 'verbose'], help='PyBuda performance trace setting')
    parser.add_argument('--netlist-name', type=str, help='Netlist name')

    # PyBuda specific placement overwrites
    parser.add_argument('--placement-overwrites', action='store_true', help='Apply placement overwites for general settings')
    parser.add_argument('--placement-overwrites-seqlen', action='store_true', help='Apply placement overwrites specific to sequence lengths')

    # Evaluation
    parser.add_argument('--do-eval', action='store_true', help='Run evaluation on all validation datasets')
    parser.add_argument('--do-mmlu-eval', action='store_true', help='Run evaluation on massive multi-task language understanding dataset')
    parser.add_argument('--max-eval-samples', type=int, default=None, help='Maximum number of samples used in evaluation set')

    # CI
    parser.add_argument('--ci', action='store_true', help='Run in CI mode')
    parser.add_argument('--ci-exit-zero', action='store_true', help='Run in CI mode but only warn if CI target metric is not met and exit with code 0. Used for debugging.')
    parser.add_argument('--ci-target', choices=['loss', 'grad_norm'], default='loss', help='CI target metric')
    parser.add_argument('--ci-tolerance', type=float, help='Assert if CI target metric (loss/grad_norm) difference between pybuda and pytorch is not within this tolerance in CI mode')
    parser.add_argument('--ci-abs-upper-limit', type=float, help='Assert if CI target metric (loss/grad_norm) not below this value in CI mode')

    # Debugging
    parser.add_argument('--data-debug', action='store_true', help='Debug dataset')


    # args = parser.parse_args()
    # When running from pytest we just ignore any arguments as we provide a config file
    args = parser.parse_args([])
    args.config = cfg_file

    # Set config file arguments if not specified explicitly on command-line
    if args.config:
        print(f'Using config {args.config}')
        with open(args.config, "r") as config_file:
            config_dict = json.load(config_file)
            args_dict = vars(args)
            for key, value in config_dict.items():
                assert key in args_dict, f"Config file contains unknown argument {key}"
                if (args_dict[key] is None or (type(args_dict[key]) is bool and not args_dict[key])):
                    args_dict[key] = value
            args = Namespace(**args_dict)


    # Set default values for arguments that are not specified explicitly
    args.learning_rate=args.learning_rate if args.learning_rate is not None else default_args.learning_rate
    args.max_grad_norm=args.max_grad_norm if args.max_grad_norm is not None else default_args.max_grad_norm
    args.data_seed=args.data_seed if args.data_seed is not None else default_args.data_seed
    args.model_seed=args.model_seed if args.model_seed is not None else default_args.model_seed
    args.sequence_length=args.sequence_length if args.sequence_length is not None else default_args.sequence_length
    args.pybuda_device=args.pybuda_device if args.pybuda_device is not None else default_args.pybuda_device
    args.num_epochs=args.num_epochs if args.num_epochs is not None else default_args.num_epochs
    args.max_num_steps=args.max_num_steps if args.max_num_steps is not None else default_args.max_num_steps
    args.num_layers=args.num_layers if args.num_layers is not None else default_args.num_layers
    args.num_lora_layers=args.num_lora_layers if args.num_lora_layers is not None else default_args.num_lora_layers
    args.rank=args.rank if args.rank is not None else default_args.rank
    args.precision=args.precision if args.precision is not None else default_args.precision
    args.version=args.version if args.version is not None else default_args.version
    args.model=args.model if args.model is not None else default_args.model
    args.lora_modules=args.lora_modules if args.lora_modules is not None else default_args.lora_modules
    args.num_chips=args.num_chips if args.num_chips is not None else default_args.num_chips
    args.batch_size=args.batch_size if args.batch_size is not None else default_args.batch_size
    args.num_accumulation_steps=args.num_accumulation_steps if args.num_accumulation_steps is not None else default_args.num_accumulation_steps
    args.loss_scale=args.loss_scale if args.loss_scale is not None else default_args.loss_scale
    args.num_samples=args.num_samples if args.num_samples is not None else default_args.num_samples
    args.dataset_name=args.dataset_name if args.dataset_name is not None else default_args.dataset_name
    args.checkpoint_every_steps=args.checkpoint_every_steps if args.checkpoint_every_steps is not None else default_args.checkpoint_every_steps
    args.checkpoint_at_steps=args.checkpoint_at_steps if args.checkpoint_at_steps is not None else default_args.checkpoint_at_steps
    args.languages=args.languages if args.languages is not None else default_args.languages
    args.prefiltered_dataset_dir=args.prefiltered_dataset_dir if args.prefiltered_dataset_dir is not None else default_args.prefiltered_dataset_dir
    args.activation_cache_file=args.activation_cache_file if args.activation_cache_file is not None else default_args.activation_cache_file
    args.num_cache_layers=args.num_cache_layers if args.num_cache_layers is not None else default_args.num_cache_layers
    args.wandb_log_steps=args.wandb_log_steps if args.wandb_log_steps is not None else default_args.wandb_log_steps
    args.wandb_project=args.wandb_project if args.wandb_project is not None else default_args.wandb_project
    args.pybuda_log_level=args.pybuda_log_level if args.pybuda_log_level is not None else default_args.pybuda_log_level
    args.optimizer_on_host=args.optimizer_on_host if args.optimizer_on_host else default_args.optimizer_on_host
    args.loss_on_device=args.loss_on_device if args.loss_on_device else default_args.loss_on_device
    args.save_checkpoints_at_epoch_end=args.save_checkpoints_at_epoch_end if args.save_checkpoints_at_epoch_end else default_args.save_checkpoints_at_epoch_end
    args.save_checkpoint_as_model_dict=args.save_checkpoint_as_model_dict if args.save_checkpoint_as_model_dict else default_args.save_checkpoint_as_model_dict
    args.save_optimizer_state=args.save_optimizer_state if args.save_optimizer_state else default_args.save_optimizer_state

    from getpass import getuser

    # Ensure a non-weka PYBUDA_BUILD_DIR for tt_build files unless specified
    pybuda_build_dir = os.environ.setdefault("PYBUDA_BUILD_DIR", f"/tmp/{getuser()}/tt_build")
    print(f"PYBUDA_BUILD_DIR set to {pybuda_build_dir}")

    # Autogenerated arguments
    lora_config_name = get_lora_short_name_from_modules(args.lora_modules)

    if args.experiment_name is None:
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.experiment_name = f'{date_time}_seqlen-{args.sequence_length}_lora-layers-{args.num_lora_layers}_lora-modules-{lora_config_name}_bs-{args.batch_size}_lr-{args.learning_rate}_rank-{args.rank}_gradnorm-{args.max_grad_norm}_acc-steps-{args.num_accumulation_steps}_precision-{args.precision}'
    experiment_path = create_experiment_dir(experiment_name=args.experiment_name)
    print(f"Output location: {experiment_path}")

    if not args.netlist_name:
        assert os.getenv('PYBUDA_NETLIST_OVERRIDE') is None, 'PYBUDA_NETLIST_OVERRIDE is set, but --netlist-name is not specified, you probably wanted to use "transformer"'
        args.netlist_name = f'seqlen-{args.sequence_length}_lora-lay-{args.num_lora_layers}_lora-mod-{lora_config_name}_r-{args.rank}'.replace('-', '_')

    if args.wandb_log_name is None:
        with_name = {(True, True): 'both', (True, False): 'pytorch', (False, True): 'pybuda'}[(args.with_pytorch, args.with_pybuda)]
        args.wandb_log_name = f'{getuser()}_with-{with_name}_{args.experiment_name}'

    print('\n\n' + '_'*20)
    print("Arguments:")
    print('\n')
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('_'*20 + '\n\n')

    parsed_args_dict = vars(args)
    parsed_args_json_path = '%s/all_args.json'%(experiment_path)
    with open(parsed_args_json_path, "w") as json_file:
        json.dump(parsed_args_dict, json_file, indent=4)


    if os.path.isdir(args.hf_cache):
        cache_dir = args.hf_cache
        # Change HF cache directory
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = cache_dir
    else:
        print("Cache folder not found. Reverting to default location")


    # Catch known unsupported cases
    if args.prefiltered_dataset_dir and not (args.load_dataset_from_disk or args.save_dataset_to_disk):
        assert False, "Dataset dir specified, but no load/store from disk. Won't have any effect!"

    if args.load_dataset_from_disk and args.dataset_name != 'guanaco_en_sp_fr':
        assert False, "Loading dataset from disk only supported for guanaco_en_sp_fr dataset."

    if args.load_dataset_from_disk and args.dataset_name == 'guanaco_en_sp_fr' and args.languages != 'en,es,fr':
        assert False, "Currently only en,sp,fr supported for guanaco and loading from disk"

    if args.explicit_pad_token:
        assert args.ignore_pad_token_loss, "When adding an explicit pad token, it should be ignored in the loss function. Set --explicit-pad-token to activate."

    assert not (args.ci_tolerance and args.ci_abs_upper_limit), 'Cannot specify both --ci-tolerance and --ci-abs-upper-limit'

    if args.ci_tolerance is not None:
        assert args.with_pytorch and args.with_pybuda, 'CI mode with a target tolerance requires both --with-pytorch and --with-pybuda'

    if args.load_in_8bit and args.version != 'padded_split':
        assert args.version == 'padded_split', "Argument load-in-8bit only supported for version=padded_split."

    if args.ci_tolerance or args.ci_abs_upper_limit:
        assert args.ci, "Need to enable --ci to use --ci-tolerance or --ci-abs-upper-limit"

    assert (args.num_lora_layers <= args.num_layers),  "Number of LoRA layers cannot be larger than number of layers"

    assert args.with_pytorch or args.with_pybuda, "Need to specify at least one trainer to run"


    if args.num_samples <= 0: # use full dataset if we specify <= 0
        args.num_samples = None

    if not args.no_wandb:
        import wandb
        os.environ['WANDB_START_METHOD']="thread"
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_log_name)

    minibatch_size = args.batch_size // args.num_accumulation_steps

    tokenizer = get_tokenizer(args.model, explicit_pad_token=args.explicit_pad_token)

    data_module = make_data_module(tokenizer, args.dataset_name, train=True, eval=True, max_train_samples=args.num_samples, max_eval_samples=args.max_eval_samples, max_seq_len=args.sequence_length,
                                   data_seed=args.data_seed, cached_dataset_dir=args.prefiltered_dataset_dir, filter_longer_sequences=args.filter_long_seq)
    train_dataloader = DataLoader(data_module["train_dataset"], batch_size=minibatch_size, collate_fn=data_module["data_collator"], shuffle=False, drop_last=True)
    eval_dataloader = DataLoader(data_module["eval_dataset"], batch_size=minibatch_size, collate_fn=data_module["data_collator"], shuffle=False, drop_last=True)
    mmlu_eval_dataloader = get_mmlu_eval_dataloader(tokenizer, args.batch_size)

    def set_model_seed():
        if args.model_seed:
            set_random_seed(args.model_seed)
        else:
            print('Warning: --model-seed set to 0, not setting a seed')

    # Create trainers as requested on command-line
    trainers = []
    if args.with_pytorch:
        set_model_seed()
        trainers.append(PyTorchTrainer(args))
    if args.with_pybuda:
        set_model_seed()
        trainers.append(PyBudaTrainer(args))
    if args.with_reference:
        set_model_seed()
        trainers.append(ReferenceTrainer('finetune_steps/1decoder/falcon_ref'))

    # Step through all trainers and show their losses side-by-side
    start_time = time()

    start_step = 0
    if args.model_load_path:
        try:
            path_name = args.model_load_path
            start_step = int(path_name.split('_')[-1].split(".")[0]) + 1
        except:
            pass

    checkpoint_step_indices = [int(idx) for idx in args.checkpoint_at_steps] if args.checkpoint_at_steps is not None else []
    print(f"Checkpointing at indicides: {checkpoint_step_indices}")

    print("\nStarting fine-tuning ...................\n")

    step = 0
    accumulation_step = 0
    num_batches = len(train_dataloader)
    last_batch_with_accumulation_per_epoch = (int(num_batches / args.num_accumulation_steps) * args.num_accumulation_steps) - 1

    for epoch in range(1, args.num_epochs+1):
        batch_index = 0
        for inputs in train_dataloader:
            if accumulation_step < start_step: # used to load from checkpoint and continue with same data order
                step += 1
                batch_index += 1
                accumulation_step = int(step / args.num_accumulation_steps)
                print(f"Skipping batch due to start_step: {start_step}")
                continue

            if args.max_num_steps > 0 and accumulation_step >= args.max_num_steps: # stop training after max_num_steps
                print(f"Reached max num accumulation steps: {args.max_num_steps}")
                break
            if batch_index > last_batch_with_accumulation_per_epoch: # ignore the last batches in the train set if they don't contribute to an optimizer update
                print(f"Skipping last {int(num_batches - (last_batch_with_accumulation_per_epoch + 1))} batches due to undivisable num accumulation steps. Batch index: {batch_index}, num batches: {num_batches}, num accumulation steps: {args.num_accumulation_steps}, epoch {epoch}")
                break

            is_first_minibatch_in_batch = (step % args.num_accumulation_steps == 0)
            is_last_minibatch_in_batch = ((step+1) % args.num_accumulation_steps == 0)
            accumulation_step = int(step / args.num_accumulation_steps)
            minibatch_in_batch = (step % args.num_accumulation_steps)+1

            losses = [ trainer.forward_step(inputs)                                                                      for trainer in trainers                    ]
            _      = [ trainer.backward_step(zero_grad=is_first_minibatch_in_batch)                                      for trainer in trainers                    ]
            _      = [ trainer.dump_step(step, args.dump_steps)                                                          for trainer in trainers if args.dump_steps ]

            _      = [ trainer.optimizer_step(do_update=is_last_minibatch_in_batch)                                      for trainer in trainers                    ]

            if not args.no_wandb and is_last_minibatch_in_batch and (accumulation_step % args.wandb_log_steps == 0):
                _  = [ trainer.log(accumulation_step)                                                                   for trainer in trainers                    ]

            trainer_output = ' | '.join(trainer.summary(includes_opt_update=is_last_minibatch_in_batch, minibatch_in_batch=minibatch_in_batch) for trainer in trainers)
            print('%5.1f Step %5d Acc-step %5d| %s' % (time() - start_time, step, accumulation_step, trainer_output))
            sys.stdout.flush()

            if is_last_minibatch_in_batch:
                _  = [ trainer.clear_batch_stats()                                                                       for trainer in trainers                      ]

            if is_last_minibatch_in_batch and ((args.checkpoint_every_steps is not None and accumulation_step % args.checkpoint_every_steps == 0 and accumulation_step != 0) or (accumulation_step in checkpoint_step_indices)):
                print(f"Saving checkpoint at accumulation step: {accumulation_step}")
                for trainer in trainers:
                    checkpoint_name = '%s/%s/ckpt_step_%d' % (experiment_path, trainer.name, accumulation_step)
                    trainer.save(checkpoint_name)

            if args.data_debug:
                pytorch_loss, pybuda_loss = losses[0], losses[1]
                is_loss_difference = abs(pytorch_loss - pybuda_loss) / pytorch_loss > 0.05
                pytorch_grad_norm, pybuda_grad_norm = trainers[0].grad_norm, trainers[1].grad_norm
                is_grad_norm_difference = abs(pytorch_grad_norm - pybuda_grad_norm) / pytorch_grad_norm > 0.05
                if is_loss_difference:
                    print(f"WARNING: Losses more than 5% apart: pytorch_loss={pytorch_loss}, pybuda_loss={pybuda_loss}")
                if is_grad_norm_difference:
                    print(f"WARNING: Gradient norms more than 5% apart: pytorch_grad={pytorch_grad_norm}, pybuda_grad={pybuda_grad_norm}")
                if is_loss_difference or is_grad_norm_difference:
                    for sample_idx, sample in enumerate(inputs['input_ids']):
                        input_text = tokenizer.decode(sample.tolist()) # , skip_special_tokens=True
                        print(f"Input text sample {sample_idx}: {input_text}")
            if args.ci_tolerance:
                if args.ci_target == 'loss':
                    pytorch_val, pybuda_val = losses[0], losses[1]
                elif args.ci_target == 'grad_norm':
                    pytorch_val, pybuda_val = trainers[0].grad_norm, trainers[1].grad_norm
                else:
                    assert False, "Target metric not supported"
                diff = abs(pytorch_val - pybuda_val)
                if diff > args.ci_tolerance:
                    error_msg = f"CI FAIL: Metric {args.ci_target} more than ci_tolerance ({args.ci_tolerance}) apart: pytorch={pytorch_val}, pybuda={pybuda_val}"

                    if args.ci_exit_zero:
                        print(error_msg)
                        exit(0)
                    else:
                        assert False, error_msg
            elif args.ci_abs_upper_limit:
                if args.ci_target == 'loss':
                    val = losses[0]
                elif args.ci_target == 'grad_norm':
                    val = trainers[0].grad_norm
                else:
                    assert False, "Target metric not supported"
                if val > args.ci_abs_upper_limit:
                    error_msg = f"CI FAIL: Metric {args.ci_target} larger than ci_abs_upper_limit ({args.ci_abs_upper_limit}): pytorch={val}"

                    if args.ci_exit_zero:
                        print(error_msg)
                        exit(0)
                    else:
                        assert False, error_msg

            step += 1
            batch_index += 1
            accumulation_step = int(step / args.num_accumulation_steps)

        if args.do_eval:
            print("Running evaluation ...................")
            results = [trainer.evaluate(eval_dataloader, mmlu_eval_dataloader)                     for trainer in trainers]
            if not args.no_wandb:
                _   = [trainer.log_eval_metrics(results[i], step, epoch)             for i, trainer in enumerate(trainers)]
            for i, trainer in enumerate(trainers):
                print(f"Evaluation Summary for {trainer.name} trainer")
                for metric, value in results[i].items():
                    print(f"{metric} : {value:.4f}")
        if args.save_checkpoints_at_epoch_end:
            print(f"Saving checkpoint at end of epoch: {epoch}")
            for trainer in trainers:
                checkpoint_name = '%s/%s/ckpt_epoch_%d' % (experiment_path, trainer.name, epoch)
                trainer.save(checkpoint_name)

    if args.save_final:
        print(f"Saving final checkpoint.")
        for trainer in trainers:
            ckpt_file_path = '%s/%s/ckpt_final' % (experiment_path, trainer.name)
            trainer.save(ckpt_file_path)
            train_log_file_path = '%s/%s/train_log.json' % (experiment_path, trainer.name)
            eval_log_file_path = '%s/%s/eval_log.json' % (experiment_path, trainer.name)
            with open(train_log_file_path, "w") as json_file:
                json.dump(trainer.train_log, json_file)
            with open(eval_log_file_path, "w") as json_file:
                json.dump(trainer.eval_log, json_file, indent=4)

    if args.save_to_activation_cache:
        for trainer in trainers:
            if args.activation_cache_out_file is None:
                if args.num_samples is None:
                    args.num_samples = 'all'
                output_file_name = f"/proj_sw/large-model-cache/falcon7b/cached_activations/cached_activations_t-{trainer.name}_l-{args.num_cache_layers}_d-{args.device}_sl-{args.seq_len}_s-{args.num_samples}.pth"
            else:
                output_file_name = args.activation_cache_out_file

            torch.save(trainer.hidden_states_cache, output_file_name)

            print(f'Saved cache as {output_file_name}')
            print(f"Number of samples saved in cache {len(trainer.hidden_states_cache.keys())}")

    if args.ci:
        print('CI PASS')

class PyTorchTrainer():
    def __init__(self, args):
        self.name = 'pytorch'
        self.args = args

        if host_device == 'cpu' and args.precision != 'fp32':
            print("WARNING: PyTorch CPU traininer currently only supported for FP32! Falling back to FP32 for PyTorch trainer.")
            args.precision = 'fp32'

        # Model
        self.tokenizer = get_tokenizer(args.model, explicit_pad_token=args.explicit_pad_token)
        self.model, _ = get_falcon_model_with_version(args.version, args.model, self.tokenizer, training=True, load_in_8bit=args.load_in_8bit,
                                                      num_layers=args.num_layers, num_cache_layers=args.num_cache_layers, load_hidden_states=args.load_from_activation_cache,
                                                      ignore_padding_loss=args.ignore_pad_token_loss, mask_lora_activations=args.mask_lora, explicit_pad_token=args.explicit_pad_token)

        if args.model_load_path and os.path.isdir(args.model_load_path):
            self.model = PeftModel.from_pretrained(self.model, args.model_load_path, is_trainable=True)
        else:
            self.model = lorify_model(self.model, target_modules=args.lora_modules, rank=args.rank, num_lora_layers=args.num_lora_layers)
            if args.model_load_path:
                checkpoint = torch.load(args.model_load_path)
                self.model.load_state_dict(checkpoint['model'])

        if args.precision == 'bf16':
            self.mixed_precision = True
            # Make frozen params to bfloat16 and trainable params as fp32
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)
                else:
                    param.data = param.data.to(torch.bfloat16)
        else:
            self.mixed_precision = False

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)


        if args.optimizer_load_state_dict:
            checkpoint = torch.load(args.optimizer_load_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])


        # model.gradient_checkpointing_enable()
        self.model.train()

        # scale loss to account for gradient accumulation
        self.loss_scale = args.loss_scale
        self.effective_loss_scale = 1.0 * args.loss_scale / args.num_accumulation_steps
        self.model.loss_fct.set_loss_scale(self.effective_loss_scale)

        self.model.to(host_device)

        self.train_log = []
        self.eval_log = []
        self.time_per_opt_step = 0.0
        self.grad_norm = 0.0
        self.accumulated_batch_loss = 0.0
        self.accumulated_batch_time_per_step = 0.0
        self.log_histograms = args.log_histograms
        self.log_opt = args.log_opt

        self.hidden_states_cache = {}


    def forward_step(self, batch):
        step_start_time = time()
        inputs = batch["input_ids"].to(host_device)
        targets = inputs.clone().detach().to(host_device)
        if self.mixed_precision:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(inputs, labels=targets, past_hidden_states=batch["hidden_states"])
        else:
            outputs = self.model(inputs, labels=targets, past_hidden_states=batch["hidden_states"])
        self.time_per_fwd_step = (time() - step_start_time)
        self.accumulated_batch_time_per_step += self.time_per_fwd_step

        self.scaled_loss = outputs.loss
        self.loss = self.scaled_loss.detach().to('cpu').item() / self.effective_loss_scale
        self.accumulated_batch_loss += self.loss

        if self.args.save_to_activation_cache:
            outputs.hidden_states = outputs.hidden_states.to('cpu')

            for input_ids, hidden_state in zip(batch["input_ids"], outputs.hidden_states):
                self.hidden_states_cache[tuple(input_ids.tolist())] = hidden_state

        return self.loss

    def backward_step(self, zero_grad):
        step_start_time = time()
        self.scaled_loss.backward()
        self.time_per_bwd_step = (time() - step_start_time)
        self.accumulated_batch_time_per_step += self.time_per_bwd_step

    def optimizer_step(self, do_update):
        if not do_update:
            self.grad_norm = 0.0
            return

        step_start_time = time()
        self.gradients = {}
        self.params = {}
        # scale back gradients
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter.grad = parameter.grad/self.loss_scale
            if self.log_histograms:
                self.gradients[name] = parameter.grad.detach().cpu().numpy().copy()
                self.params[name] = parameter.data.detach().cpu().numpy().copy()

        self.grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm))
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.time_per_opt_step = (time() - step_start_time)
        self.accumulated_batch_time_per_step += self.time_per_opt_step

    def clear_batch_stats(self):
        self.accumulated_batch_loss = 0.0
        self.accumulated_batch_time_per_step = 0.0

    def summary(self, includes_opt_update, minibatch_in_batch):
        time_per_step = self.time_per_fwd_step + self.time_per_bwd_step
        if includes_opt_update:
            time_per_step += self.time_per_opt_step
        return '%s: %.8f loss, %.8f avg batch loss %.8f grad norm in %.2f ms' % (self.name, self.loss, self.accumulated_batch_loss/minibatch_in_batch, self.grad_norm, time_per_step * 1000)

    def save(self, filename):
        if self.args.save_checkpoint_as_model_dict:
            if self.args.save_optimizer_state:
                checkpoint = {"model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict()}
            else:
                checkpoint = {"model": self.model.state_dict()}
            torch.save(checkpoint,
                       filename + ".pt")
        else:
            self.model.save_pretrained(filename) # Save only the LoRA adapter weights
            if self.args.save_optimizer_state:
                checkpoint = {"optimizer": self.optimizer.state_dict()}
                torch.save(checkpoint,
                        filename + ".pt")


    def log(self, step):
        log_dict = {
            f"{self.name}/time_per_step": self.accumulated_batch_time_per_step,
            f"{self.name}/batch_loss": self.accumulated_batch_loss / self.args.num_accumulation_steps,
            f"{self.name}/grad_norm": self.grad_norm,
            f"{self.name}/step": step,
        }
        self.train_log.append(log_dict)

        wandb.log(log_dict, step=step)

        if self.log_histograms:
            hist_dict = {}
            for param_name in self.params.keys():
                hist_dict[f"{self.name}/grads/{param_name}"] = wandb.Histogram(self.gradients[param_name])
                hist_dict[f"{self.name}/params/{param_name}"] = wandb.Histogram(self.params[param_name])

            wandb.log(hist_dict, step=step)

        if self.log_opt:
            opt_state = self.optimizer.state_dict()
            opt_log_dict = {}
            for param_name in opt_state.keys():
                opt_log_dict[f"{self.name}/opt/{param_name}"] = opt_state[param_name]
            wandb.log(opt_log_dict, step=step)

    def log_eval_metrics(self, metrics, step, epoch):
        log_dict = {}
        metrics_to_log = ['val_loss', 'mmlu_loss', 'mmlu_eval_accuracy']
        for metrics_key in metrics_to_log:
            if metrics_key in metrics:
                log_dict[f"{self.name}/{metrics_key}"] = metrics[metrics_key]

        log_dict[f"{self.name}/epoch"] = epoch

        self.eval_log.append(log_dict)

        wandb.log(log_dict, step=step)

    def dump_step(self, step, basename):
        filename = '%s-%s-%d.pt' % (basename, self.name, step)
        state_dict = { k: v      for k, v in self.model.state_dict()       if v.requires_grad }
        grad_dict =  { k: v.grad for k, v in self.model.named_parameters() if v.requires_grad }
        opt_dict = self.optimizer.state_dict()
        torch.save({'state': state_dict,
                    'grad' : grad_dict,
                    'opt'  : opt_dict},
                   filename)

    def mmlu_evaluate(self, mmlu_eval_dataloader):
        IGNORE_INDEX = self.tokenizer.pad_token_id
        abcd_idx = [
            self.tokenizer("A", add_special_tokens=False).input_ids[0],
            self.tokenizer("B", add_special_tokens=False).input_ids[0],
            self.tokenizer("C", add_special_tokens=False).input_ids[0],
            self.tokenizer("D", add_special_tokens=False).input_ids[0]]

        scores = {'refs':[], 'preds':[]}
        loss_mmlu = 0
        for batch in tqdm(mmlu_eval_dataloader, total=len(mmlu_eval_dataloader)):
            inputs = batch["input_ids"].to(host_device)
            targets = batch["labels"].to(host_device)
            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model(inputs, labels=targets, past_hidden_states=None)
                else:
                    outputs = self.model(inputs, labels=targets, past_hidden_states=None)
            loss = outputs.loss.detach().to('cpu') / self.effective_loss_scale
            logits = outputs.logits
            labels = batch["labels"]
            # There are two tokens, the output, and eos token.

            for i, logit in enumerate(logits):
                label_non_zero_id = (labels[i] != IGNORE_INDEX).nonzero()[0][0]
                logit_abcd = logit[label_non_zero_id-1][abcd_idx]

                scores['preds'].append(torch.argmax(logit_abcd).item())
                scores['refs'].append(abcd_idx.index(labels[i][label_non_zero_id]))
            loss_mmlu += loss.item()
        # Extract results by subject.
        results = {'mmlu_loss':loss_mmlu/len(mmlu_eval_dataloader)}

        accuracy = evaluate.load("accuracy")

        results[f'mmlu_eval_accuracy'] = accuracy.compute(references=scores['refs'], predictions=scores['preds'])['accuracy']

        return results

    def evaluate(self, eval_dataloader, mmlu_eval_dataloader=None):
        metrics = {}
        self.model.eval()
        val_loss = 0
        for inputs in eval_dataloader:
            with torch.no_grad():
                val_loss+=self.forward_step(inputs)
        metrics['val_loss'] = val_loss/len(eval_dataloader)

        if self.args.do_mmlu_eval:
            mmlu_metrics = self.mmlu_evaluate(mmlu_eval_dataloader)
            metrics.update(mmlu_metrics)

        self.model.train() #Turning on training mode
        return metrics

class PyBudaTrainer():
    def __init__(self, args):
        self.name = 'pybuda'

        self.args = args

        try:
            netlist_override_file = os.environ['PYBUDA_NETLIST_OVERRIDE']
            print("WARNING: PYBUDA_NETLIST_OVERRIDE is set, overriding netlist with %s" % netlist_override_file)
        except KeyError:
            pass

        os.environ["LOGGER_LEVEL"] = args.pybuda_log_level
        os.environ["LOGURU_LEVEL"] = args.pybuda_log_level
        # pybuda workarounds
        os.environ["GOLDEN_WORMHOLE_B0"] = "1"
        os.environ["WHA0_DISABLE_RELAY_BUFS"] = "1"
        os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
        os.environ["PYBUDA_CONVERT_PARAMS_TO_TVM"] = "0"
        os.environ["TT_BACKEND_TIMEOUT"] = "0"

        # lora specific
        os.environ['TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE'] = "147456" # for AdamW: "294912" invalid grid # "245760" oom # "196608" oom # "147456" oom
        os.environ['PYBUDA_DRAM_FLIP_FLOP'] = '1'

        # os.environ['PYBUDA_DRAM_PICK_CAPACITY'] = '1' # Don't need this anymore and better for perf without

        # # Testing for better handling out of valid grid issues with seqlen=1024
        # os.environ["PYBUDA_PADDING_PASS"] = "1"
        # os.environ["PYBUDA_PADDING_PASS_MATMUL"] = "1"


        pybuda = self.pybuda = __import__('pybuda') # let us set log levels before importing pybuda


        # Model
        tokenizer = get_tokenizer(args.model, explicit_pad_token=args.explicit_pad_token)
        self.model, _ = get_falcon_model_with_version(args.version, args.model, tokenizer, training=True, load_in_8bit=args.load_in_8bit,
                                                      num_layers=args.num_layers, num_cache_layers=args.num_cache_layers, load_hidden_states=args.load_from_activation_cache,
                                                      ignore_padding_loss=args.ignore_pad_token_loss, mask_lora_activations=args.mask_lora, explicit_pad_token=args.explicit_pad_token)

        if args.model_load_path and os.path.isdir(args.model_load_path):
            # self.model = lorify_model(self.model, target_modules=args.lora_modules, rank=args.rank, num_lora_layers=args.num_lora_layers)
            self.model = PeftModel.from_pretrained(self.model, args.model_load_path)
        else:
            self.model = lorify_model(self.model, target_modules=args.lora_modules, rank=args.rank, num_lora_layers=args.num_lora_layers)
            if args.model_load_path:
                checkpoint = torch.load(args.model_load_path)
                self.model.load_state_dict(checkpoint['model'])

        if args.optimizer_on_host:
            self.optimizer = torch.optim.AdamW(self.model.transformer.parameters(), lr=args.learning_rate)
        else:
            self.optimizer = pybuda.optimizers.AdamW(learning_rate=args.learning_rate, device_params=True)

        if args.optimizer_load_state_dict:
            checkpoint = torch.load(args.optimizer_load_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.model.train()

        # scale loss to account for gradient accumulation
        self.loss_scale = args.loss_scale
        self.effective_loss_scale = 1.0 * args.loss_scale / args.num_accumulation_steps
        self.model.loss_fct.set_loss_scale(self.effective_loss_scale)
        self.train_log = []
        self.eval_log = []
        self.hidden_states_cache = {}

        self.log_histograms = args.log_histograms
        self.log_opt = args.log_opt
        device = args.pybuda_device

        if args.tt_build_dir:
            backend_output_dir = os.path.join(pybuda_build_dir, args.tt_build_dir)
            os.environ["PYBUDA_BUILD_DIR"] = backend_output_dir 
            print("Setting backend output dir to %s" % backend_output_dir)
            pybuda.set_configuration_options(backend_output_dir=backend_output_dir)

        if device == 'golden':
            devtype = pybuda.BackendType.Golden
        elif device == 'silicon':
            devtype = pybuda.BackendType.Silicon
        else:
            raise NotImplementedError('Unknown device "%s" is not supported' % device)

        self.embeddings_module = pybuda.PyTorchModule("embeddings", self.model.before_decoders)
        self.transformer_module = pybuda.PyTorchModule(args.netlist_name, self.model.transformer)
        self.lm_head_module = pybuda.PyTorchModule("lm_head", self.model.after_decoders)
        self.loss_module = pybuda.PyTorchModule("loss", self.model.loss_fct)

        if args.precision == 'fp32':
            default_df_override = pybuda.DataFormat.Float32
        elif args.precision == 'fp16':
            default_df_override = pybuda.DataFormat.Float16
        elif args.precision == 'bf16':
            default_df_override = pybuda.DataFormat.Float16_b
        elif args.precision == 'very-low-mp':
            default_df_override = pybuda.DataFormat.Float16_b
        elif args.precision == 'almost-low-mp':
            default_df_override = pybuda.DataFormat.Float16_b
        elif args.precision == 'low-mp':
            default_df_override = pybuda.DataFormat.Float16_b
        elif args.precision == 'high-mp':
            default_df_override = pybuda.DataFormat.Float16_b
        elif args.precision == 'debug':
            default_df_override = pybuda.DataFormat.Float16_b
        else:
            default_df_override = None

        num_layers = args.num_layers - args.num_cache_layers if args.load_from_activation_cache else args.num_layers

        self.apply_data_formats(lora_data_format=pybuda.DataFormat.Float32, num_layers=num_layers, num_lora_layers=args.num_lora_layers, precision=args.precision)
        self.apply_custom_overrides()
        if args.placement_overwrites:
            self.placement_overwrites(num_layers=num_layers, num_lora_layers=args.num_lora_layers)
        if args.placement_overwrites_seqlen:
            self.placement_overwrites_seqlen(num_layers=num_layers, num_lora_layers=args.num_lora_layers)
        if args.num_chips > 1:
            self.multichip_placement(num_layers=num_layers, num_lora_layers=args.num_lora_layers, precision=args.precision)

        perf_level = { None    : None,
                        'none'   : None,
                        'light'  : pybuda.PerfTraceLevel.LIGHT,
                        'verbose': pybuda.PerfTraceLevel.VERBOSE }[args.perf]

        pybuda.set_configuration_options(
                                        default_df_override=default_df_override,
                                         accumulate_df=pybuda.DataFormat.Float32,
                                         amp_level=0,
                                         enable_auto_fusing=False,
                                         performance_trace=perf_level,
                                         backend_opt_level=4,
                                         enable_auto_transposing_placement=True,
                                        #  backend_cluster_descriptor_path="/proj_sw/user_dev/jrock/pybuda-falcon-stable-avx/pybuda/third_party/budabackend/wormhole_2chip_cluster.yaml" if args.num_chips > 1 else None,
                                         )
        pybuda.config._get_global_compiler_config().use_interactive_placer = True

        self.cpu0 = pybuda.CPUDevice("cpu0", module=self.embeddings_module)

        start_time = time()
        if args.tti_load is not None:
            print(f"Load TTImage from : {args.tti_load}")
            self.tt0 = pybuda.TTDevice.load_image(img_path=args.tti_load)
        else:
            self.tt0 = pybuda.TTDevice('tt0', module=self.transformer_module,
                                        arch=pybuda.BackendDevice.Wormhole_B0,
                                        devtype=devtype,
                                        chip_ids=list(range(args.num_chips)),
                                        optimizer=self.optimizer if not args.optimizer_on_host else None)
        duration = (time() - start_time)
        print(f"Time to create tt device / load tti: {duration}")

        if args.loss_on_device:
            self.tt0.place_module(self.lm_head_module)
            self.tt0.place_loss_module(self.loss_module)
        else:
            self.cpu1 = pybuda.CPUDevice("cpu1", module=self.lm_head_module)
            self.cpu1.place_loss_module(self.loss_module)

        mp = torch.multiprocessing.get_context('spawn')
        self.output_q = mp.Queue()

        if args.verify:
            self.verify_cfg=pybuda.VerifyConfig(enabled=True,
                                                # waive_gradient_errors=small_bias_keys, # errors are invalid for small values
                                                scale_loss=1.0, # defaults to 50!
                                                golden_ignore_df_precision=True,
                                                test_kind=self.pybuda.verify.config.TestKind.TRAINING,
                                                devtype=pybuda.BackendType.Silicon,
                                                arch=pybuda.BackendDevice.Wormhole_B0,
                                                pcc=99.999,
                                                verify_all=True,
                                                verify_last=True
                                                )

        else:
            self.verify_cfg = None
        self.initialized = False

        self.time_per_opt_step = 0.0
        self.grad_norm = 0.0
        self.accumulated_batch_loss = 0.0
        self.accumulated_batch_time_per_step = 0.0

    def apply_custom_overrides(self):

        pass

    def placement_overwrites(self, num_layers, num_lora_layers):
        for i in range(10000): # Overwrite all gelu op placements
            self.pybuda.config.override_op_size(f'bw_in0_gelu_{i}_multiply_1', (1, 8))
        self.pybuda.config.override_op_size('input_1_multiply_247_splt_brcst_1_0_splt_brcst_3_0', (1, 4))


    def placement_overwrites_seqlen(self, num_layers, num_lora_layers):

        # Resource limits exceeded for the following cores
        # Core (c=0,y=4,x=6) [routing]  (c=0,y=3,x=4) [worker] [op_name=bw_in1_matmul_83_matmul_1] exceeded resource constraints:
        # active dram queues used: 68 limit: 40

        lora_loop_offset = 92

        # lora
        for i in range(num_lora_layers):
            # bwd
            # bw_in0_matmul_87_matmul_1
            # bw_in0_matmul_271_matmul_1
            op_name = f'bw_in0_matmul_{87 + lora_loop_offset*i}_matmul_1'
            self.pybuda.config.override_op_size(op_name, (4, 8))
            print("Setting op size ", op_name, " to (4, 8)")

            # TODO: check if we need this!
            op_name = f'bw_in1_matmul_{83 + lora_loop_offset*i}_matmul_1'
            self.pybuda.config.override_op_size(op_name, (8, 2))
            print("Setting op size ", op_name, " to (8, 2)")

            # For > 1 layers
            op_name = f'bw_in0_gelu_{22 + lora_loop_offset*i}_gelu_derivative_0'
            self.pybuda.config.override_op_size(op_name, (1, 8))
            print("Setting op size ", op_name, " to (1, 8)")

            op_name = f'bw_in0_matmul_{29 + lora_loop_offset*i}_matmul_1'
            self.pybuda.config.override_op_size(op_name, (2, 8))
            print("Setting op size ", op_name, " to (1, 8)")


    def multichip_placement(self, num_layers, num_lora_layers, precision):
        chip_id_0 = 1
        chip_id_1 = 0

        non_lora_loop_offset_multiply = 70
        non_lora_loop_offset_matmul = 70

        lora_loop_offset_multiply = 92
        lora_loop_offset_matmul = 92


        num_non_lora_layers = num_layers - num_lora_layers

        # Add chip breaks at layer boundaries

        # non-lora fwd
        for i in range(num_non_lora_layers):
            op_name = f"multiply_{0 + non_lora_loop_offset_multiply*i}"
            self.pybuda.config.set_epoch_break(op_name)
            print("Setting epoch break at ", op_name)

        initial_offset_multiply = num_non_lora_layers*non_lora_loop_offset_multiply
        initial_offset_matmul = num_non_lora_layers*non_lora_loop_offset_matmul
        # lora
        for i in range(num_lora_layers):
            # fwd
            op_name = f"multiply_{0 + initial_offset_multiply + lora_loop_offset_multiply*i}"
            self.pybuda.config.set_epoch_break(op_name)
            print("Setting epoch break at ", op_name)

            # bwd
            op_name = f"bw_in0_matmul_{87 + initial_offset_matmul + lora_loop_offset_matmul*i}_matmul_1"
            self.pybuda.config.set_epoch_break(op_name)
            print("Setting epoch break at ", op_name)




        if num_non_lora_layers == 0:
            # # set chip ids: halve layers on chip other halve on chip 1

            start_layer_idx_chip_1 = int(num_lora_layers // 2)

            # chip ids for fwd
            for i in range(num_lora_layers):
                chip_id = chip_id_1 if i >= start_layer_idx_chip_1 else chip_id_0
                op_name = f"multiply_{0 + lora_loop_offset_multiply*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id)
                print("Setting chip_id for ", op_name, " to ", chip_id)

                op_name = f"matmul_{18 + lora_loop_offset_matmul*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id)
                print("Setting chip_id for ", op_name, " to ", chip_id)

                op_name = f"matmul_{61 + lora_loop_offset_matmul*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id)
                print("Setting chip_id for ", op_name, " to ", chip_id)


            # chip ids for bwd
            for i in range(num_lora_layers):
                chip_id = chip_id_1 if i >= start_layer_idx_chip_1 else chip_id_0
                # bw_in0_matmul_157_matmul_1
                op_name = f"bw_in0_matmul_{87 + lora_loop_offset_matmul*i}_matmul_1"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id)
                print("Setting chip_id for ", op_name, " to ", chip_id)

                # bw_in0_multiply_108_multiply_0
                op_name = f"bw_in0_multiply_{38 + lora_loop_offset_multiply*i}_multiply_0"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id)
                print("Setting chip_id for ", op_name, " to ", chip_id)
        else:
            # set chip ids: non-lora on chip 0 and lora on chip 1

            # chip ids for fwd
            for i in range(num_non_lora_layers):
                op_name = f"multiply_{0 + non_lora_loop_offset_multiply*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_0)
                print("Setting chip_id for ", op_name, " to ", chip_id_0)

                op_name = f"matmul_{18 + non_lora_loop_offset_matmul*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_0)
                print("Setting chip_id for ", op_name, " to ", chip_id_0)

                op_name = f"matmul_{61 + non_lora_loop_offset_matmul*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_0)
                print("Setting chip_id for ", op_name, " to ", chip_id_0)

            initial_offset_multiply = num_non_lora_layers*non_lora_loop_offset_multiply
            initial_offset_matmul = num_non_lora_layers*non_lora_loop_offset_matmul
            for i in range(num_lora_layers):
                op_name = f"multiply_{0 + initial_offset_multiply + lora_loop_offset_multiply*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_1)
                print("Setting chip_id for ", op_name, " to ", chip_id_1)

                op_name = f"matmul_{18 + initial_offset_matmul + lora_loop_offset_matmul*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_1)
                print("Setting chip_id for ", op_name, " to ", chip_id_1)

                op_name = f"matmul_{61 + initial_offset_matmul + lora_loop_offset_matmul*i}"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_1)
                print("Setting chip_id for ", op_name, " to ", chip_id_1)


            # chip ids for bwd
            initial_offset_matmul = num_non_lora_layers*non_lora_loop_offset_matmul
            initial_offset_multiply = num_non_lora_layers*non_lora_loop_offset_multiply
            for i in range(num_lora_layers):
                # bw_in0_matmul_157_matmul_1
                op_name = f"bw_in0_matmul_{87 + initial_offset_matmul + lora_loop_offset_matmul*i}_matmul_1"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_1)
                print("Setting chip_id for ", op_name, " to ", chip_id_1)

                # bw_in0_multiply_108_multiply_0
                op_name = f"bw_in0_multiply_{38 + initial_offset_multiply + lora_loop_offset_multiply*i}_multiply_0"
                self.pybuda.config.override_op_placement(op_name, chip_id=chip_id_1)
                print("Setting chip_id for ", op_name, " to ", chip_id_1)


        # # additional hacks to fix net2pipe issues

        # Resource limits exceeded for the following cores
        # Core (c=0,y=1,x=1) [routing]  (c=0,y=0,x=0) [worker] [op_name=add_159] exceeded resource constraints:
        # active dram queues used: 44 limit: 40
        add_name = f"add_{89 + num_non_lora_layers*non_lora_loop_offset_matmul + (num_lora_layers-1)*lora_loop_offset_matmul}"
        self.pybuda.config.override_op_size(add_name, (1, 2))




    def apply_data_formats(self, lora_data_format, num_layers, num_lora_layers, precision):

        non_lora_loop_offset = 70
        lora_loop_offset = 92


        num_non_lora_layers = num_layers - num_lora_layers

        if precision == 'very-low-mp': # all matmul inputs are bfp8_b; keep bwd/loss/lora bf16b
            self.pybuda.config.configure_mixed_precision(
                    name_regex="matmul_.*",
                    input_df={0: [self.pybuda.DataFormat.Bfp8_b, True], 1: [self.pybuda.DataFormat.Bfp8_b, True], 2: [self.pybuda.DataFormat.Bfp8_b, True]})

            if not self.args.optimizer_on_host: # Overwrites needed only for opt on device
                self.pybuda.config.configure_mixed_precision(
                        name_regex=".*lora_.*",
                        output_df=self.pybuda.DataFormat.Float16_b)

        if precision == 'almost-low-mp': # all matmul inputs are bfp8_b; keep bwd/loss/lora bf16b
            self.pybuda.config.configure_mixed_precision(
                    name_regex="matmul_.*",
                    input_df={0: [self.pybuda.DataFormat.Bfp8_b, True], 1: [self.pybuda.DataFormat.Bfp8_b, True], 2: [self.pybuda.DataFormat.Bfp8_b, True]})

            self.pybuda.config.configure_mixed_precision(
                    name_regex=".*lora_.*",
                    output_df=self.pybuda.DataFormat.Float16_b,
                    input_df={0: [self.pybuda.DataFormat.Float16_b, True], 1: [self.pybuda.DataFormat.Float16_b, True], 2: [self.pybuda.DataFormat.Float16_b, True]})

            self.pybuda.config.configure_mixed_precision(
                    name_regex="loss_.*",
                    accumulate_df=lora_data_format,
                    output_df=lora_data_format)

            self.pybuda.config.configure_mixed_precision(
                    name_regex="bw_.*",
                    accumulate_df=lora_data_format,
                    output_df=lora_data_format)

        elif precision == 'low-mp': # all matmul inputs are bfp8_b except for matmul_33 (compile issue); increase bw/loss/lora to fp32
            self.pybuda.config.configure_mixed_precision(
                    name_regex="matmul_.*",
                    input_df={0: [self.pybuda.DataFormat.Bfp8_b, True], 1: [self.pybuda.DataFormat.Bfp8_b, True], 2: [self.pybuda.DataFormat.Bfp8_b, True]})

            # except all lora ops + params
            self.pybuda.config.configure_mixed_precision(
                    name_regex=".*lora_.*",
                    output_df=lora_data_format,
                    input_df={0: [lora_data_format, True], 1: [lora_data_format, True], 2: [lora_data_format, True]})

            self.pybuda.config.configure_mixed_precision(
                    name_regex="loss_.*",
                    accumulate_df=lora_data_format,
                    output_df=lora_data_format)

            self.pybuda.config.configure_mixed_precision(
                    name_regex="bw_.*",
                    accumulate_df=lora_data_format,
                    output_df=lora_data_format)

        elif precision == 'high-mp': # all falcon frozen weights are bfp8_b, everything else is bfp16_b

            # all frozen weights
            self.pybuda.config.configure_mixed_precision(
                    name_regex="input_0_transpose_.*",
                    output_df=self.pybuda.DataFormat.Bfp8_b,
                    )

            # except all lora ops + params
            self.pybuda.config.configure_mixed_precision(
                    name_regex=".*lora_.*",
                    output_df=lora_data_format,
                    input_df={0: [lora_data_format, True], 1: [lora_data_format, True], 2: [lora_data_format, True]})

            self.pybuda.config.configure_mixed_precision(
                    name_regex="loss_.*",
                    accumulate_df=lora_data_format,
                    output_df=lora_data_format)

            self.pybuda.config.configure_mixed_precision(
                    name_regex="bw_.*",
                    accumulate_df=lora_data_format,
                    output_df=lora_data_format)

            # required for falcon-stable pybuda (08/2023)
            # <PIPEGEN-ERROR> OP matmul_664 packer : Chip = 0, Core x = 18, y = 19(logical x = 0, y = 1): out of memory. Total bytes alloced = 1295424 B, used by bufs = 977920 B, used by pipegen = 317504 B Limit: 1290240
            # non-lora
            for i in range(num_non_lora_layers):
                op_name = f"matmul_{20 + non_lora_loop_offset*i}"
                self.pybuda.config.override_op_size(op_name, (4, 8))
                print("Overwriting  op size for ", op_name, "to (4, 8)")

            initial_offset = num_non_lora_layers*non_lora_loop_offset
            # lora
            for i in range(num_lora_layers):
                op_name = f"matmul_{20 + initial_offset + lora_loop_offset*i}"
                self.pybuda.config.override_op_size(op_name, (4, 8))
                print("Overwriting  op size for ", op_name, "to (4, 8)")

        elif precision == 'debug': # use this for debugging wihout changing existing options
            pass


    def ensure_initialized(self, batch):
        if not self.initialized:
            start_time = time()
            if self.args.tti_save is not None:
                self.tt0.compile_to_image(
                    img_path=self.args.tti_save,
                    training=True,
                    sample_inputs=batch,
                    sample_targets=batch,
                    microbatch_count=1
                )
                print(f'Saved TTImage to {self.args.tti_save}')
                sys.exit(0)
            self.pybuda.initialize_pipeline(training=True,
                                    sample_inputs=batch,
                                    sample_targets=batch,
                                    output_queue=self.output_q,
                                    # microbatch_count=self.args.num_accumulation_steps will require buffering for all elements in batch in fwd->bwd e2e queues!
                                    microbatch_count=1,
                                    _sequential=True,
                                    _verify_cfg=self.verify_cfg,
                                    )
            duration = (time() - start_time)
            print(f"Time to save image / initialize pipeline: {duration}")
        self.initialized = True

    def forward_step(self, batch):

        if self.args.load_from_activation_cache:
            inputs = (batch["input_ids"], batch["hidden_states"])
        else:
            inputs = (batch["input_ids"],)

        targets = (batch["input_ids"].clone(),)

        self.ensure_initialized(inputs)

        step_start_time = time()
        self.cpu0.push_to_inputs(inputs)

        if self.args.loss_on_device:
            self.tt0.push_to_target_inputs(targets)
        else:
            self.cpu1.push_to_target_inputs(targets)

        self.pybuda.run_forward(input_count=1, _sequential=True)
        self.time_per_fwd_step = (time() - step_start_time)
        self.accumulated_batch_time_per_step += self.time_per_fwd_step

        self.scaled_loss = self.output_q.get(timeout=60)[0]
        self.loss = self.scaled_loss / self.effective_loss_scale
        self.accumulated_batch_loss += self.loss

        assert not self.args.save_to_activation_cache, "Saving activation cache for pybuda model not implemented yet!"

        return self.loss

    def backward_step(self, zero_grad):
        step_start_time = time()
        self.pybuda.run_backward(input_count=1, zero_grad=zero_grad, _sequential=True)
        self.time_per_bwd_step = (time() - step_start_time)
        self.accumulated_batch_time_per_step += self.time_per_bwd_step

    def optimizer_step(self, do_update):
        if not do_update and self.args.optimizer_on_host:
            self.pybuda.sync()
        if not do_update:
            self.grad_norm = 0.0
            return

        step_start_time = time()

        if self.args.optimizer_on_host:
            tt_device = self.tt0
            module = self.model.transformer

            # Get gradients from chip
            gradients = self.pybuda.get_parameter_gradients(tt_device, _sequential=True)[0]

            self.gradients = {}
            self.params = {}
            new_device_parameters = dict()

            # Assign gradients to module parameters
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue

                param_grad = gradients[name].value()
                parameter.grad = param_grad.type(parameter.dtype)

                # scale back gradients
                parameter.grad = parameter.grad/self.loss_scale

                if self.log_histograms:
                    self.gradients[name] = parameter.grad.detach().cpu().numpy().copy()
                    self.params[name] = parameter.data.detach().cpu().numpy().copy()

            # clip gradient norm
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), self.args.max_grad_norm))

            # No need to self.optimizer.zero_grad() because we are overwriting the gradients anyway
            self.optimizer.step()

            # Collecting new device parameters in dictionary (update trainable parameters only)
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                new_device_parameters[name] = parameter

            # Send updated weights back to device
            tt_device.update_device_parameters(new_device_parameters)
        else:
            #FIXME: add gradient clipping for optimizer on device!
            #FIXME: add gradient scaling for optimizer on device!

            # if self.args.extensive_logging:
            #     self.gradients = self.pybuda.get_parameter_gradients(self.tt0, _sequential=True)[0]
            #     self.grad_norm = get_gradient_norm(self.model.transformer.named_parameters(), self.gradients)
            #     self.params = self.pybuda.get_parameter_checkpoint(self.tt0, _sequential=True)[0]

            self.pybuda.run_optimizer(_sequential=True)
            if self.log_histograms:
                # FIXME: not tested. get gradients and params from device for logging
                self.gradients = self.pybuda.get_parameter_gradients(tt_device, _sequential=True)[0]
                self.params = self.pybuda.get_parameters(tt_device, _sequential=True)[0]

        self.time_per_opt_step = (time() - step_start_time)
        self.accumulated_batch_time_per_step += self.time_per_opt_step

    def clear_batch_stats(self):
        self.accumulated_batch_loss = 0.0
        self.accumulated_batch_time_per_step = 0.0

    def summary(self, includes_opt_update, minibatch_in_batch):
        time_per_step = self.time_per_fwd_step + self.time_per_bwd_step
        if includes_opt_update:
            time_per_step += self.time_per_opt_step
        return '%s: %.8f loss, %.8f avg batch loss, %.8f grad norm in %.2f ms' % (self.name, self.loss, self.accumulated_batch_loss/minibatch_in_batch, self.grad_norm, time_per_step*1000)

    def log(self, step):
        log_dict = {
            f"{self.name}/time_per_step": self.accumulated_batch_time_per_step,
            f"{self.name}/batch_loss": self.accumulated_batch_loss / self.args.num_accumulation_steps,
            f"{self.name}/grad_norm": self.grad_norm,
        }

        self.train_log.append(log_dict)

        wandb.log(log_dict, step=step)

        if self.log_histograms:
            hist_dict = {}
            for param_name in self.params.keys():
                hist_dict[f"{self.name}/grads/{param_name}"] = wandb.Histogram(self.gradients[param_name])
                hist_dict[f"{self.name}/params/{param_name}"] = wandb.Histogram(self.params[param_name])

            wandb.log(hist_dict, step=step)

        if self.log_opt:
            opt_state = self.optimizer.state_dict()
            opt_log_dict = {}
            for param_name in opt_state.keys():
                opt_log_dict[f"{self.name}/opt/{param_name}"] = opt_state[param_name]
            wandb.log(opt_log_dict, step=step)

        if step == 1:
            try:
                netlist_path = self.args.netlist_name + "_netlist.yaml"
                artifact = wandb.Artifact("netlist", type="yaml")
                artifact.add_file(netlist_path)
                wandb.log_artifact(artifact)
            except:
                print(f"WARNING: Could not log netlist at path: {netlist_path}")

            try:
                netlist_override_file = os.environ['PYBUDA_NETLIST_OVERRIDE']
                artifact = wandb.Artifact("netlist_override", type="yaml")
                artifact.add_file(netlist_override_file)
                wandb.log_artifact(artifact)
            except KeyError:
                pass

    def log_eval_metrics(self, metrics, step, epoch):
        metrics_to_log = ['val_loss']
        log_dict = {
            f"{self.name}/{metrics_key}": metrics[metrics_key] for metrics_key in metrics_to_log
        }
        log_dict[f"{self.name}/epoch"] = epoch

        self.eval_log.append(log_dict)

        wandb.log(log_dict, step=step)

    def save(self, filename):
        if not self.args.optimizer_on_host:
            module = self.model.transformer

            # Get parameters from device
            device_params = self.pybuda.get_parameter_checkpoint(self.tt0, _sequential=True)[0]

            # Assign gradients to module parameters
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue

                device_param = device_params[name].value()
                parameter.data = device_param.type(parameter.data.dtype)
        
        if self.args.save_checkpoint_as_model_dict:
            if self.args.save_optimizer_state:
                checkpoint = {"model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict()}
            else:
                checkpoint = {"model": self.model.state_dict()}
            torch.save(checkpoint,
                       filename + ".pt")
        else:
            self.model.save_pretrained(filename) # Save only the LoRA adapter weights
            if self.args.save_optimizer_state:
                checkpoint = {"optimizer": self.optimizer.state_dict()}
                torch.save(checkpoint,
                        filename + ".pt")


    def evaluate(self, eval_dataloader, mmlu_eval_dataloader=None):
        metrics = {}
        val_loss = 0
        for inputs in eval_dataloader:
            val_loss+=self.forward_step(inputs)
            self.backward_step(False)
        metrics['val_loss'] = val_loss/len(eval_dataloader)
        return metrics


class ReferenceTrainer:
    def __init__(self, base_filename):
        self.name = 'reference'
        self.filename = lambda step: base_filename + '_%d.pt' % step
        self.step = 0
        self.loss = None

    def forward_step(self, batch):
        self.step += 1
        data = torch.load(self.filename(self.step), map_location=torch.device('cpu'))

        assert (batch['input_ids'] == data['input_ids']).all(), 'Training dataloader does not match the reference batch in step %d' % self.step
        self.loss = data['loss']
        return self.loss

    def backward_step(self, zero_grad):
        pass

    def optimizer_step(self):
        pass

    def summary(self):
        return '%s: %.8f loss' % (self.name, self.loss)

    def log(self, step):
        pass

    def save(self, filename):
        pass
