# XGLM Demo - CausalLM
import os

import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, XGLMConfig, XGLMForCausalLM


def run_xglm_causal_lm(variant="facebook/xglm-564M"):

    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("take")
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_enumerate_u_kt = False

    # Variants: "facebook/xglm-564M", "facebook/xglm-1.7B"
    model_ckpt = variant
    if model_ckpt == "facebook/xglm-1.7B":
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        compiler_cfg.amp_level = 1

    # set model configurations
    config = XGLMConfig.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = XGLMConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    model = XGLMForCausalLM.from_pretrained(model_ckpt, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"

    # Create text generator object
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Run inference on Tenstorrent device
    answer = text_generator(
        prefix_text,
        max_length=20,
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
    run_xglm_causal_lm()
