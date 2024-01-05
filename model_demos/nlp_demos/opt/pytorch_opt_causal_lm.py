# OPT demo script - CausalLM

import os

import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM


def run_opt_casual_lm(variant="facebook/opt-350m"):

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    model_ckpt = variant

    if model_ckpt == "facebook/opt-1.3b" or model_ckpt == "facebook/opt-350m":
        compiler_cfg.enable_auto_fusing = False
        if model_ckpt == "facebook/opt-1.3b":
            compiler_cfg.amp_level = 2

            # Disable expanding output buffer of fork nodes - causes out of memory issue in blobgen.
            os.environ["PYBUDA_FORK_JOIN_EXPAND_FORK_OUTPUT_BUF"] = "0"
        if variant == "facebook/opt-350m":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"

    # Set model configurations
    config = OPTConfig.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = OPTConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    model = OPTForCausalLM.from_pretrained(model_ckpt, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"

    # Run inference on Tenstorrent device
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    answer = text_generator(
        prefix_text,
        max_length=30,
        num_beams=4,
        num_return_sequences=2,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    print(f"Prefix text: {prefix_text}")
    print("Generated text:")
    for sequence in answer:
        print(sequence.values())


if __name__ == "__main__":
    run_opt_casual_lm()
