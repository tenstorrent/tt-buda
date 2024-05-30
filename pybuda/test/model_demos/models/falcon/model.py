# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Model class for Falcon LLM.
"""

import os

import pybuda
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .configuration_RW import RWConfig
from .pybudify import PyBudify
from .tt_modeling_RW_pad_masked_odkv import (
    RWForCausalLM as RWForCausalLMPaddedMaskedODKV,
)

MODEL_CKPT = "tiiuae/falcon-7b-instruct"
MAX_SEQUENCE_LENGTH = os.environ.get("MAX_SEQUENCE_LENGTH", 2048)
NUM_TOKENS = os.environ.get("NUM_TOKENS", 100)
TOP_P_ENABLE = os.environ.get("TOP_P_ENABLE", 1)
TOP_P = os.environ.get("TOP_P", 0.9)
TOP_K = os.environ.get("TOP_K", 40.0)
TEMPERATURE = os.environ.get("TEMPERATURE", 1.0)


params = {
    "alibi": False,
    "apply_residual_connection_post_layernorm": False,
    "architectures": ["RWForCausalLM"],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_RW.RWConfig",
        "AutoModel": "modelling_RW.RWModel",
        "AutoModelForSequenceClassification": "modelling_RW.RWForSequenceClassification",
        "AutoModelForTokenClassification": "modelling_RW.RWForTokenClassification",
        "AutoModelForQuestionAnswering": "modelling_RW.RWForQuestionAnswering",
        "AutoModelForCausalLM": "modelling_RW.RWForCausalLM",
    },
    "bias": False,
    "bos_token_id": 11,
    "eos_token_id": 11,
    "hidden_dropout": 0.0,
    "hidden_size": 4544,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "RefinedWebModel",
    "multi_query": True,
    "n_head": 71,
    "padded_heads": 1,
    "n_layer": 32,
    "parallel_attn": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.27.4",
    "use_cache": True,
    "vocab_size": 65024,
}


class Falcon:
    def __init__(
        self,
        user_rows=32,
        num_tokens=int(NUM_TOKENS),
        max_length=int(MAX_SEQUENCE_LENGTH),
        top_p_enable=int(TOP_P_ENABLE),
        top_p=float(TOP_P),
        top_k=float(TOP_K),
        temperature=float(TEMPERATURE),
        model_ckpt=MODEL_CKPT,
    ):
        self.batch_size = 1
        self.user_rows = user_rows
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.layers = 32
        self.top_p_enable = top_p_enable
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_ckpt = model_ckpt

    def initialize(self):

        pybuda.set_configuration_options(
            backend_output_dir="tt_build/decode_demo"
        )

        config = RWConfig(**params, user_rows=self.user_rows)
        config.n_layer = self.layers
        self.model = RWForCausalLMPaddedMaskedODKV.from_pretrained(
            MODEL_CKPT, config=config
        )
        self.model.transformer.split_qkv_weights()
        self.model.transformer.pad_decoders()  # After loading weights, pad the decoders

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
        self.tokenizer.pad_token_id = 0  # '[PAD]' - set pad token for tokenizer

        self.model.eval()

        netlist_name = "falcon_1c_0mf_0af_32l_2048s"
        self.model.transformer.blocks = PyBudify(
            self.model.transformer.blocks,
            device="silicon",
            arch="wormhole_b0",
            precision="bf16",
            amp_level=0,
            num_chips=1,
            fuse=False,
            perf=None,
            verify=False,
            log_level="ERROR",
            tti_load=None,
            tti_save=None,
            concurrent=False,
            fracture=False,
            padded_fracture=False,
            padded_fracture_p=False,
            padded_fracture_full=False,
            odkv=False,
            masked_odkv=True,
            fracture_attn=0,
            fracture_mlp=0,
            netlist_name=netlist_name,
            host_queues=False,
            num_layers=self.layers,
        )

    def inference(self, prompts):
        top_ps = []
        temperatures = []
        n_requests = len(prompts)
        assert 1 <= n_requests <= self.user_rows, (
            "n_requests is not in the range 1-user_rows!"
            f"n_requests = {n_requests}, user_rows = {self.user_rows}"
        )
        prompts += ["False start"] * (self.user_rows - n_requests)
        top_ps += [self.top_p] * self.user_rows
        temperatures += [self.temperature] * self.user_rows
        top_ps = torch.tensor(top_ps).unsqueeze(1)
        temperatures = torch.tensor(temperatures).unsqueeze(1)
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )

        tokenized_input_ids = tokenized.input_ids.unsqueeze(0).clone()
        tokenized_attention_mask = tokenized.attention_mask.unsqueeze(0).clone()
        prompt_token_counts = [sum(mask) for mask in tokenized.attention_mask]

        # initial attention mask, rows will be overwritten for prefill users
        # attention_mask = torch.ones((args.batch_size, args.user_rows, args.seqlen), dtype=torch.long, device=tokenized_attention_mask.device)
        # initial attention mask, rows will be overwritten for prefill users
        attention_mask = torch.zeros(
            (self.batch_size, self.user_rows, self.max_length), dtype=torch.long
        )
        input_ids = tokenized_input_ids[:, :, 0].clone()

        assert (
            min(prompt_token_counts) > 0
        ), "Empty prompts for unconditional generation not currently supported"
        assert (
            self.batch_size == 1
        ), "Pretty sure this code assumes batch size == 1, FIXME"

        # tensor of right size and shape needed for pybuda to compile. initialise kv with zeros
        # value in tensor doesn't matter. we're going to prefill this in anyways
        # TODO: replace constants 32 and 64
        past_key_values = tuple(
            [
                (
                    torch.zeros((self.batch_size, 1, self.max_length, 64 * 32)),
                    torch.zeros((self.batch_size, 1, self.max_length, 64 * 32)),
                )
                for _ in range(self.layers)
            ]
        )

        # initial wait for keypress will be removed from latencies as warmup time at end
        num_tokens = 0
        all_output = [
            self.tokenizer.decode(
                input_ids[0, i],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            ).replace("\n", " ")
            for i in range(self.user_rows)
        ]

        end_token_pos = [None for _ in range(self.user_rows)]

        # Prepare attention_mask and position_ids for decode mode. We will override these for users still in prefill mode.
        # in decode phase we pay attention to all new tokens so concat 1 to attention_mask for the latest token
        # and shift out oldest tokens attention mask FIFO style similar to odkv cache logic
        # attention_mask = torch.cat((attention_mask, torch.ones((1, args.user_rows, 1))), dim=2)
        # attention_mask = attention_mask[:, :, 1:]

        while True:
            if self.num_tokens and num_tokens >= self.num_tokens:
                break

            # Now override any attention and input rows for users that are still prefilling from their prompt
            for i in range(self.user_rows):
                # in in prefill phase for any user, pick attention_mask from tokenized_attention_mask
                if num_tokens < prompt_token_counts[i]:
                    # at the very least we have 1 prefill token so seqlen - 1 are the unused tokens. and then subtract num_tokens as we prefill them
                    # mask out tokens which haven't been prefilled
                    # attention_mask[:, i, :args.seqlen - num_tokens - 1] = 0

                    attention_mask[:, i, num_tokens] = tokenized_attention_mask[
                        :, i, num_tokens
                    ]

                    # we get and set the prefill tokens attention mask according to the tokeniser which will mask out padded tokens for us
                    # attention_mask[:, i, args.seqlen - num_tokens - 1:] = tokenized_attention_mask[:, i, :num_tokens + 1]
                    # attention_mask[:, i, :num_tokens+1] = tokenized_attention_mask[:, i, :num_tokens+1]

                    # prefill mode picks input_ids from the prompt tokens
                    input_ids[:, i] = tokenized_input_ids[:, i, num_tokens]

                else:  # prefill is done for this user so just set attention mask to 1 for next token while rolling over
                    attention_mask[:, i, num_tokens % self.max_length] = 1

            # As we use right-padding all users have the same num_tokens
            position_ids = torch.tensor([[num_tokens]], dtype=torch.long)
            position_ids = position_ids.expand(
                self.batch_size, self.user_rows
            ).clone()

            kv_read_mask = torch.ones((1, 1, self.max_length, 1))
            kv_read_mask[:, :, [num_tokens % self.max_length], :] = 0

            kv_write_mask = torch.zeros((1, 1, self.max_length, 1))
            kv_write_mask[:, :, [num_tokens % self.max_length], :] = 1

            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    kv_read_mask=kv_read_mask,
                    kv_write_mask=kv_write_mask,
                )
            logits = outputs[0].to(
                "cpu"
            )  # FIXME: doesn't this assume args.batch_size == 1?
            logits /= self.temperature

            if not self.top_p_enable:  # greedy
                token = logits.argmax(dim=-1)
            else:
                token = sample_kp_logits(logits, self.top_k, self.top_p)

            # Use the expected output from the prompt as long as we are still prefilling; only the last token is from the model
            for i in range(self.user_rows):
                if num_tokens < prompt_token_counts[i] - 1:
                    token[i] = tokenized_input_ids[0, i, num_tokens + 1]

            for i in range(self.user_rows):
                if (
                    end_token_pos[i] is None
                    and token[i] == self.tokenizer.eos_token_id
                ):
                    end_token_pos[i] = num_tokens

            for i in range(self.user_rows):
                if end_token_pos[i] is None:
                    all_output[i] += self.tokenizer.decode(
                        token[i],
                        clean_up_tokenization_spaces=True,
                        skip_special_tokens=True,
                    )

            # if we're at the end of prefilling use the newly-generated token as input (overridden above if we are prefilling)
            input_ids = token.unsqueeze(
                0
            )  # FIXME: doesn't this assume args.batch_size == 1?

            # Update the new and total token counts
            num_tokens += 1

        return all_output


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_kp_logits(logits, k, p):
    next_token_logscores = top_k_top_p_filtering(logits, top_k=k, top_p=p)
    probs = F.softmax(next_token_logscores, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token
