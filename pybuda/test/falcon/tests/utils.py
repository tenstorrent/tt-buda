# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
""" Shared utility functions for working with falcon models """
import torch
import os
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
from langdetect import detect, LangDetectException
from functools import partial
import random
import transformers
from dataclasses import dataclass
import copy
import urllib.request
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence
import json

from ..models.falcon7b.tt_modeling_RW import RWForCausalLM
from ..models.falcon7b.tt_modeling_RW_pad import RWForCausalLM as RWForCausalLMPadded
from ..models.falcon7b.tt_modeling_RW_pad_split import RWForCausalLM as RWForCausalLMPaddedSplit
from ..models.falcon7b.tt_modeling_RW_pad_split_cache import RWForCausalLM as RWForCausalLMPaddedSplitCache
from ..models.falcon7b.modelling_RW import RWForCausalLM as RWForCausalLMTorch1
from ..models.falcon7b.modelling_RW_original import RWForCausalLM as RWForCausalLMTorch2
from ..models.falcon7b.configuration_RW import RWConfig


ds_wiki_test = None


def set_random_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  # CuBLAS and CUDA >= 10.2. require this to use deterministic behaviour, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

def get_lora_short_name_from_modules(lora_modules):
    # Mapping of lora_modules to names

    lora_modules_sorted = sorted(lora_modules)

    lora_modules_names = {
        ('wq', 'wv'): 'qv',
        ('wk', 'wq', 'wv'): 'qkv',
        ('dense', 'wk', 'wq', 'wv'): 'attention',
        ('dense_4h_to_h', 'dense_h_to_4h'): 'mlp',
        ('dense', 'dense_4h_to_h', 'dense_h_to_4h', 'wk', 'wq', 'wv'): 'all'
    }
    return lora_modules_names[tuple(lora_modules_sorted)]

def get_gradient_norm(named_params, gradients):
    total_norm = 0.0
    for name, parameter in named_params:
        if not parameter.requires_grad:
            continue

        param_grad = gradients[name].value()
        param_grad_norm = param_grad.data.norm(2)
        total_norm += param_grad_norm.item() ** 2
    grad_norm = float(total_norm ** 0.5)
    return grad_norm


def get_tokenizer(model_name, explicit_pad_token = False, hf_cache=None):
    # Get tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)
    if explicit_pad_token:
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Num new tokens added : {num_new_tokens}")
    else:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


def get_tokenized_data_loaders(tokenizer, context_length, batch_size, num_samples,
                               dataset_name="timdettmers/openassistant-guanaco", languages=None,
                               load_prefiltered_dataset_from_disk=True, prefiltered_dataset_dir=None,
                               save_prefiltered_dataset_to_disk=True,
                               cache_file=None, load_hidden_states=False,
                               data_seed=None, preprocess_data=False):
    
    if load_prefiltered_dataset_from_disk:
        ds_train_dir = prefiltered_dataset_dir + "_train"
        ds_test_dir = prefiltered_dataset_dir + "_test"
        assert os.path.exists(ds_train_dir), f"Filtered train dataset doesn't exist at path: {ds_train_dir}"
        assert os.path.exists(ds_test_dir), f"Filtered test dataset doesn't exist at path: {ds_test_dir}"
        ds_train =  load_from_disk(ds_train_dir)
        ds_test = load_from_disk(ds_test_dir)
        print(f"Loading prefiltered dataset from {prefiltered_dataset_dir}")
        print(f'Train: {len(ds_train)} samples')
        print(f'Test : {len(ds_test)} samples')
    else:
        ds_train = load_dataset(dataset_name, split="train")
        ds_test = load_dataset(dataset_name, split="test")

        if languages is not None:
            def check_lang(example):
                try:
                    return detect(example['text']) in languages
                except LangDetectException:
                    return False

            orig_len_train = len(ds_train)
            orig_len_test = len(ds_test)
            ds_train = ds_train.filter(check_lang)
            ds_test = ds_test.filter(check_lang)
            print(f'Train: filtered {orig_len_train} -> {len(ds_train)} samples for languages: {languages}')
            print(f'Test : filtered {orig_len_test} -> {len(ds_test)} samples for languages: {languages}')
            assert len(ds_train) > 0, "No training samples found for languages: " + str(languages)
            assert len(ds_test) > 0, "No test samples found for languages: " + str(languages)

            if save_prefiltered_dataset_to_disk:
                ds_train_dir = prefiltered_dataset_dir + "_train"
                ds_test_dir = prefiltered_dataset_dir + "_test"
                ds_train.save_to_disk(ds_train_dir)
                ds_test.save_to_disk(ds_test_dir)
                print(f"Saved prefiltered dataset to: {prefiltered_dataset_dir}")

    if num_samples is not None:
        print(f"Using reduced dataset with {num_samples} samples.")
        if num_samples < len(ds_train):
            ds_train = ds_train.select(range(num_samples))
        if num_samples < len(ds_test):
            ds_test = ds_test.select(range(num_samples))

    # Ensure deterministic ordering by shuffling the dataset once and then using the same shuffle for all runs
    # Note: using shuffle=True in the dataloader is lazy so the random number generator can be disturbed by other calls
    if data_seed is not None:
        set_random_seed(data_seed)
        ds_train = ds_train.shuffle(seed=data_seed)
    else:
        print('Warning: --data-seed set to 0, not setting a seed')
        ds_train = ds_train.shuffle()

    raw_datasets = DatasetDict(
        {
            "train": ds_train,
            "test": ds_test,
        }
    )

    def preprocess(chats):
        """
        Add the eos_token after each response, so during inference the model can stop generating tokens after it completes its response. For example
        ### HUMAN:
        Hello
        ### Assistant:
        Hi, how are you?<eos_token>
        ### HUMAN:
        I'm fine.
        ### Assistant:
        How can I help you?<eos_token>
        """

        processed_chats = []
        for chat in chats:
            splitted_chat = chat.split("### Human:")
            for i in range(1,len(splitted_chat)):
                splitted_chat[i] = splitted_chat[i] + tokenizer.eos_token
            processed_chat = "### Human:".join(splitted_chat)
            processed_chats.append(processed_chat)
        return processed_chats

    def tokenize(samples):
        tokenized_samples = tokenizer(
            preprocess(samples["text"]) if preprocess_data else samples["text"],
            truncation=True,
            padding=True,
            max_length=context_length
        )
        return tokenized_samples

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    
    hidden_states_cache = None
    if load_hidden_states:
        assert cache_file is not None, "Cache file is None"
        hidden_states_cache = torch.load(cache_file)
        print(f"Hidden States Cache Loaded Successfully from {cache_file}")

    def collate_fn(data, cache=None):
            inputs={
                "input_ids": torch.stack([item['input_ids'].type(torch.long) for item in data]),
                "token_type_ids": torch.stack([item['token_type_ids'].type(torch.long) for item in data]),
                "attention_mask": torch.stack([item['attention_mask'].type(torch.long) for item in data]),
                "hidden_states": torch.stack([cache[tuple(item['input_ids'].tolist())] for item in data]) if cache else None
            }
            return inputs
    
    collate_fn_with_cache = partial(collate_fn, cache=hidden_states_cache)
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_with_cache)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_with_cache)
    
    return train_dataloader, eval_dataloader

def get_falcon_model_with_version(version, model_name, tokenizer, training, num_layers=None, load_in_8bit=False, hidden_size=None, 
                                  cache_hidden_states=False, num_cache_layers=None, load_hidden_states=False, ignore_padding_loss=False, 
                                  mask_lora_activations=False, explicit_pad_token=False, hf_cache=None):
    # Download the config
    config = RWConfig.from_pretrained(model_name, user_rows=32, cache_dir=hf_cache)
    config.use_cache = not training
    if num_layers is not None:
        config.n_layer = num_layers

    config.bos_token_id=tokenizer.bos_token_id
    config.eos_token_id=tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    assert not ignore_padding_loss or version == "padded_split", "Ignore padding loss only supported for padded_split model"
    if explicit_pad_token:
        assert ignore_padding_loss, "Ignore padding in loss, if explicit padding token is added"

    if cache_hidden_states or load_hidden_states:
        version = 'padded_split_cache'

    if version == 'huggingface':
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, trust_remote_code=True, device_map={"":0}, cache_dir=hf_cache)
    elif version == 'fractured':
        model = RWForCausalLM.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, cache_dir=hf_cache)
        model.transformer.split_qkv_weights() # necessary to pre-split qkv
        model.transformer.split_layernorm_weights()
    elif version == 'padded':
        config.padded_heads = 1 # pad up to 72 heads
        model = RWForCausalLMPadded.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, cache_dir=hf_cache)
        model.transformer.split_qkv_weights() # necessary to pre-split qkv
        model.transformer.pad_decoders() # After loading weights, pad the decoders
    elif version == 'padded_split':
        config.ignore_pad_tokens = ignore_padding_loss
        config.padded_heads = 1 # pad up to 72 heads
        config.mask_lora_activations = mask_lora_activations #Apply masking for lora activations
        model = RWForCausalLMPaddedSplit.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, cache_dir=hf_cache)
        if explicit_pad_token:
            model.resize_token_embeddings(len(tokenizer)) # Resize pretrained embedding by concatenating new tokens embeddings(random weights)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            #Intialize padding i/o embeddings with zeros
            input_embeddings[-1] = torch.zeros(input_embeddings.size(-1))
            output_embeddings[-1] = torch.zeros(output_embeddings.size(-1))
            #Modify resized embedding references
            model.before_decoders.word_embeddings = model.transformer.word_embeddings 
        model.transformer.split_qkv_weights() # necessary to pre-split qkv
        model.transformer.pad_decoders() # After loading weights, pad the decoders
    elif version == 'padded_split_cache':
        config.ignore_pad_tokens = ignore_padding_loss
        config.padded_heads = 1 # pad up to 72 heads
        config.cache_hidden_states = cache_hidden_states
        config.num_cache_layers = num_cache_layers
        config.load_hidden_states = load_hidden_states
        model = RWForCausalLMPaddedSplitCache.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, cache_dir=hf_cache)
        if explicit_pad_token:
            model.resize_token_embeddings(len(tokenizer)) # Resize pretrained embedding by concatenating new tokens embeddings(random weights)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            #Intialize padding i/o embeddings with zeros
            input_embeddings[-1] = torch.zeros(input_embeddings.size(-1))
            output_embeddings[-1] = torch.zeros(output_embeddings.size(-1))
            #Modify resized embedding references
            model.before_decoders.word_embeddings = model.transformer.word_embeddings
        model.transformer.split_qkv_weights() # necessary to pre-split qkv
        model.transformer.pad_decoders() # After loading weights, pad the decoders
    elif version == 'padded_split_tiny': # pretrained weights won't load
        config.padded_heads = 1
        config.n_head = 31
        config.hidden_size = 496
        print(f"Using padded_split tiny model with hidden size as {config.hidden_size}")
        model = RWForCausalLMPaddedSplit(config=config)
        model.transformer.split_qkv_weights() # necessary to pre-split qkv
        model.transformer.pad_decoders() # After loading weights, pad the decoders
    elif version == 'torch1.0':
        model = RWForCausalLMTorch1.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, cache_dir=hf_cache)
    elif version == 'torch2.0':
        model = RWForCausalLMTorch2.from_pretrained(model_name, config=config, load_in_8bit=load_in_8bit, cache_dir=hf_cache)
    else:
        raise ValueError(f"Unknown version {version}")

    return model, config
    


def lorify_model(model, target_modules, rank, num_lora_layers=None):
    # prepare_model_for_kbit_training is not needed at the moment, sets everything to FP32 and we overwrite data formats in pybuda anyways
    #     from peft import prepare_model_for_kbit_training
    #     model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
            # print(f"Using FP32 for module name {name}")

    num_layers = model.config.n_layer
    if num_lora_layers is None:
        num_lora_layers = num_layers

    if num_lora_layers == 0: # don't use lora at all
        return model

    lora_target_modules = []
    parent_module = ""
    for module_name in target_modules:
        if module_name in {'wq', 'wk', 'wv', 'dense', 'query_key_value'}:
            parent_module = "self_attention"
        elif module_name in {"dense_h_to_4h", "dense_4h_to_h"}:
            parent_module = "mlp"
        else:
            raise Exception(f"Lora module {module_name} not found!")
        
        # Add last num_lora_layers layers to lora_target_modules
        for lora_layer in range(num_layers-num_lora_layers, num_layers):
            lora_target_modules.append(str(lora_layer)+ "." + parent_module + "." + module_name)
    
    print(f"Adding lora layers with rank: {rank} to target modules: {lora_target_modules}")

    # Config LoRA and get model
    # r=64, alpha=16 as in QLoRA paper
    # r=16, alpha=32 as in a random HF model / colab: dfurman/falcon-7b-openassistant-peft 
    # r=32, alpha=8 (ours) (in LoRA paper they claim adjusting alpha is = adjusting learning rate, so we keep ratio the same as in QLoRA)
    config = LoraConfig(
        lora_alpha=0.25*rank,
        lora_dropout=0.01,
        r=rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules
    )

    model = get_peft_model(model, config)
    # print_trainable_parameters(model)
    verify_datatypes(model)
    # print(model)
    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.1f}"
    )


# Verify the datatypes, expected: frozen part is int8, trainable is fp32
def verify_datatypes(model):
    dtypes = {}
    for name, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)


def eval_model_on_text_corpus(model, dataset, tokenizer, max_length=512, stride=512):
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    model.eval()

    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    mean_nll = torch.stack(nlls).mean()
    ppl = torch.exp(mean_nll)
    return mean_nll.item(), ppl.item()


def eval_model(model, dataloader, device):
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            targets = inputs.clone()
            outputs = model(inputs, labels=targets)

        losses.append(outputs.loss)
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def get_grouped_params(model, weight_decay=0.1, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def token_loss(logits, inputs):
    # Shift so that tokens < n predict n
    shift_labels = inputs[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def eval_model_all_datasets(model, train_dataloader, eval_dataloader, tokenizer, device):
    global ds_wiki_test
    model.eval()

    print("Model evaluation")
    
    # train_loss, train_ppl = eval_model(model, train_dataloader, device)
    # print({"loss/train": train_loss, "perplexity/train": train_ppl})

    test_loss, test_ppl = eval_model(model, eval_dataloader, device)
    print({"loss/eval": test_loss, "perplexity/eval": test_ppl})

    # if ds_wiki_test is None:
    #     # ds_wiki_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=hf_cache)
    #     ds_wiki_test = load_dataset("wikitext", split="test", cache_dir=hf_cache)
    # wiki_loss, wiki_ppl = eval_model_on_text_corpus(model, ds_wiki_test, tokenizer)
    # print({"loss/wiki": wiki_loss, "perplexity/wiki": wiki_ppl})

    # test_inference_prompt(model, tokenizer)


def test_inference_prompt(model, tokenizer):
    prompt = """### Human: Can you please provide me the names of the two players in the atomic bomb game (in go)? If you can get me the referee's name as well, that's even better!### Assistant:"""

    batch = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    batch = batch.to(device)

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids = batch.input_ids, 
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # print(generated_text.split("### Human: ")[1].split("### Assistant: ")[-1])
    print(" * * * Test prompt: * * * ")
    print(prompt)
    print(" * * * Test response: * * * ")
    print(generated_text)

@dataclass
class DataCollatorForCausalLM(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        
        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            hidden_states=None
        )
        
def get_mmlu_dataset(max_mmlu_samples=100, tokenizer=None, hf_cache=None):
    data_files = {'eval': 'mmlu/five_shot_mmlu_val.json', 'test': 'mmlu/five_shot_mmlu_test.json'}
    url_map = {
        'eval': 'https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/five_shot_mmlu_val.json',
        'test': 'https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/five_shot_mmlu_test.json'
    }

    # Create the directory if it doesn't exist
    if not os.path.exists('mmlu'):
        os.makedirs('mmlu')

    # Check and download JSON files if they don't exist
    for split, file_name in data_files.items():
        if not os.path.exists(file_name):
            print(f"'{file_name}' doesn't exist. Downloading...")
            try:
                urllib.request.urlretrieve(url_map[split], file_name)
                print(f"'{file_name}' downloaded successfully.")
            except Exception as e:
                print(f"Error downloading '{file_name}': {e}")

    mmlu_dataset = load_dataset("json", data_files=data_files, cache_dir=hf_cache)
    mmlu_dataset = mmlu_dataset['eval']
    if max_mmlu_samples is not None:
        mmlu_dataset = mmlu_dataset.select(range(max_mmlu_samples))
    
    new_mmlu_dataset = []
    max_token_count = 384

    for row in mmlu_dataset:
        input_text = row["input"]
        output_text = row["output"]
        
        # Tokenize the input text
        input_tokens = tokenizer.tokenize(input_text)
        
        # Check if token count is within limits
        if len(input_tokens) <= max_token_count:
            new_row = {
                "subject": row["subject"],
                "input": input_text,
                "output": output_text
            }
            new_mmlu_dataset.append(new_row)

    return new_mmlu_dataset

def get_mmlu_eval_dataloader(tokenizer, batch_size, filter_longer_sequences=True, max_seq_len=384):
    
    mmlu_eval_data_module = make_data_module(tokenizer, "mmlu_eval", train=False, eval=True, filter_longer_sequences=filter_longer_sequences, max_seq_len=max_seq_len)
    return DataLoader(mmlu_eval_data_module["eval_dataset"], batch_size=batch_size, collate_fn=mmlu_eval_data_module["data_collator"], shuffle=False, drop_last=True)


def create_experiment_dir(parent_dir='falcon_lora_experiments', experiment_name='untitled'):
    os.makedirs(parent_dir, exist_ok=True)
    experiment_path = os.path.join(parent_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    pybuda_exp_path = os.path.join(experiment_path, 'pybuda')
    os.makedirs(pybuda_exp_path, exist_ok=True)
    pytorch_exp_path = os.path.join(experiment_path, 'pytorch')
    os.makedirs(pytorch_exp_path, exist_ok=True)
    return experiment_path


def get_dataset(dataset_name, cached_dataset_dir=None, hf_cache=None):
        if dataset_name == 'alpaca':
            dataset = load_dataset("tatsu-lab/alpaca", cache_dir=hf_cache)
        elif dataset_name == 'alpaca_eval':
            dataset = load_dataset("tatsu-lab/alpaca_eval", download_mode="force_redownload", cache_dir=hf_cache)
        elif dataset_name == 'guanaco':
            dataset = load_dataset("timdettmers/openassistant-guanaco", cache_dir=hf_cache)
        elif dataset_name == 'guanaco_en_sp_fr':
            ds_dir = os.path.join(cached_dataset_dir, "openassistant_guanaco_en_sp_fr")
            if os.path.exists(ds_dir):
                print(f"Loading prefiltered dataset from {ds_dir}")
                dataset =  load_from_disk(ds_dir)
                print(f'Train: {len(dataset["train"])} samples')
                print(f'Test : {len(dataset["test"])} samples')
            else:
                print(f"Filtered guanaco_en_sp_fr dataset doesn't exist")
                dataset = load_dataset("timdettmers/openassistant-guanaco", cache_dir=hf_cache)
                languages = ['en','es','fr']
                def check_lang(example):
                    try:
                        return detect(example['text']) in languages
                    except LangDetectException:
                        return False

                orig_len_train = len(dataset["train"])
                orig_len_test = len(dataset["test"])
                dataset = dataset.filter(check_lang)
                print(f'Train: filtered {orig_len_train} -> {len(dataset["train"])} samples for languages: {languages}')
                print(f'Test : filtered {orig_len_test} -> {len(dataset["test"])} samples for languages: {languages}')                
                dataset.save_to_disk(ds_dir)
                print(f"Saved prefiltered dataset to: {ds_dir}")

        elif dataset_name == 'mmlu_eval':
            # Create the directory if it doesn't exist
            os.makedirs('mmlu', exist_ok=True)
            data_files = {'eval': 'mmlu/five_shot_mmlu_val.json', 'test': 'mmlu/five_shot_mmlu_test.json'}
            url_map = {
                'eval': 'https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/five_shot_mmlu_val.json',
                'test': 'https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/five_shot_mmlu_test.json'
            }
            
            # Check and download JSON files if they don't exist
            for split, file_name in data_files.items():
                if not os.path.exists(file_name):
                    print(f"'{file_name}' doesn't exist. Downloading...")
                    try:
                        urllib.request.urlretrieve(url_map[split], file_name)
                        print(f"'{file_name}' downloaded successfully.")
                    except Exception as e:
                        print(f"Error downloading '{file_name}': {e}")
            dataset = load_dataset("json", data_files=data_files, cache_dir=hf_cache)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
        
        return dataset

def format_dataset(dataset, dataset_name):
    if dataset_name == 'alpaca':
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        dataset = dataset.remove_columns([col for col in dataset.column_names['train'] if col not in ['input', 'output']])
    
    elif dataset_name == 'alpaca_eval':
        dataset = dataset.map(format_alpaca_eval_dataset)

    elif dataset_name == 'guanaco' or dataset_name == 'guanaco_en_sp_fr':
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['text'],
        })
        dataset = dataset.remove_columns([col for col in dataset.column_names['train'] if col not in ['input', 'output']])

    elif dataset_name == 'mmlu_eval':
        # leave as is
        dataset = dataset.remove_columns([col for col in dataset.column_names['eval'] if col not in ['input', 'output']])
    # Remove unused columns.
    
    return dataset



def preprocess_dataset(dataset, dataset_name, tokenizer, filter_longer_sequences=True, filter_longer_inputs=False, max_input_len=256, max_seq_len=512):

    def append_eos(text, dataset_name):
        if dataset_name == 'guanaco' or dataset_name == 'guanaco_en_sp_fr':
            """
            Add the eos_token after each response, so during inference the model can stop generating tokens after it completes its response. For example
            ### HUMAN:
            Hello
            ### Assistant:
            Hi, how are you?<eos_token>
            ### HUMAN:
            I'm fine.
            ### Assistant:
            How can I help you?<eos_token>
            """

            splitted_text = text.split("### Human:")
            for i in range(1,len(splitted_text)):
                splitted_text[i] = splitted_text[i] + tokenizer.eos_token
            processed_text = "### Human:".join(splitted_text)
        
        else:
            processed_text = text + tokenizer.eos_token

        return processed_text
    
    dataset = dataset.map(lambda x: {
                            'output': append_eos(x['output'], dataset_name),
                        })
    
    def filter_long_sequences(sample):
        complete_sequence = sample['input'] + sample['output']
        tokenized_complete_sequence = tokenizer.tokenize(complete_sequence)
        return len(tokenized_complete_sequence)<=max_seq_len
    
    def filter_long_inputs(sample, max_input_len=max_input_len):
        tokenized_input = tokenizer.tokenize(sample['input'])
        return len(tokenized_input)<=max_input_len
    
    if "eval" in dataset:
        orig_len_eval = len(dataset["eval"])
    if "train" in dataset:
        orig_len_train = len(dataset["train"])
    if filter_longer_sequences:
        dataset = dataset.filter(filter_long_sequences)
    if filter_longer_inputs:
        dataset = dataset.filter(filter_long_inputs)
    if "train" in dataset:
        print(f'Train: filtered {orig_len_train} -> {len(dataset["train"])} samples for max_seq_len: {max_seq_len}')
    if "eval" in dataset:
        print(f'Eval: filtered {orig_len_eval} -> {len(dataset["eval"])} samples for max_seq_len: {max_seq_len}')
    return dataset

def extract_alpaca_dataset(example):
    ALPACA_PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    }
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def format_alpaca_eval_dataset(example):
    prompt_format ="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n### Human: {instruction}\n### Assistant:"
    return {'input': prompt_format.format(**example)}

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, dataset_name, train=True, eval=True, 
                     max_train_samples=None, max_eval_samples=None, max_seq_len=512, data_seed=42, cached_dataset_dir=None, hf_cache=None,
                     filter_longer_sequences=True, filter_longer_inputs=False, max_input_len=256, train_on_source=False) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    """
    raw_dataset = get_dataset(dataset_name, cached_dataset_dir, hf_cache)
    formatted_dataset = format_dataset(raw_dataset, dataset_name)
    dataset = preprocess_dataset(formatted_dataset, dataset_name, tokenizer, filter_longer_sequences, filter_longer_inputs, max_input_len, max_seq_len)
    
    def tokenize(dataset):
        """Preprocess the data by tokenizing."""
        sources = dataset['input']
        targets = dataset['output']
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized = tokenizer(
                                    examples,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=max_seq_len
                                )
        sources_tokenized = tokenizer(
                                    sources,
                                    truncation=True,
                                    padding=False,
                                    max_length=max_seq_len
                                )
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        #Don't train on source
        for label, source in zip(labels, sources_tokenized["input_ids"]):
            label[:len(source)] = [tokenizer.pad_token_id for _ in range(len(source))]
        return dict(input_ids=input_ids, labels=labels, attention_mask=examples_tokenized["attention_mask"])

    # Split train/eval, reduce size
    if eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            print('Splitting train dataset in train and validation according to `90-10`')
            dataset = dataset["train"].train_test_split(
                test_size=0.10, shuffle=True, seed=data_seed
            )
            eval_dataset = dataset['test']
        
        if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # At the moment do not remove the columns
        # eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
        eval_dataset = eval_dataset.map(tokenize, batched=True)

        print(f'Eval Dataset Length : {len(eval_dataset)}')

    if train:
        train_dataset = dataset['train']
        if max_train_samples is not None and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        if data_seed is not None:
            set_random_seed(data_seed)
            train_dataset = train_dataset.shuffle(seed=data_seed)
        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)

        print(f'Train Dataset Length : {len(train_dataset)}')

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    return dict(
            train_dataset = train_dataset if train else None,
            eval_dataset = eval_dataset if eval else None,
            data_collator = data_collator)

def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
