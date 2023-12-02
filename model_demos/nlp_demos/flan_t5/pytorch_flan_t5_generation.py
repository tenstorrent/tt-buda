# FLAN-T5 Demo - Conditional Generation

import os

import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config


def run_flan_t5_pybuda_pipeline(variant="google/flan-t5-small"):

    # Add PyBUDA configurations
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "30000"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_amp_light()

    # Variants: google/flan-t5-small, google/flan-t5-base, google/flan-t5-large
    model_ckpt = variant

    # Set model configurations
    config = T5Config.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = T5Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Sample prompt
    prefix_text = "A step by step recipe to make bolognese pasta:"

    # Initialize text2text generator
    text2text_generator = pybuda_pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        pybuda_max_length=32,
    )

    # Run inference on Tenstorrent device
    answer = text2text_generator(
        prefix_text,
        max_length=20,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    # Report output
    print(f"Prefix text: {prefix_text}")
    print("Generated text:")
    for sequence in answer:
        print(sequence.values())


if __name__ == "__main__":
    run_flan_t5_pybuda_pipeline()
