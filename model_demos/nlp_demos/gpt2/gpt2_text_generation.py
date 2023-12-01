# GPT2 demo script - Text generation
import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


def run_gpt2_text_gen():

    # Set model configurations
    config = GPT2Config.from_pretrained("gpt2")
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPT2Config(**config_dict)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Load tokenizer and model from HuggingFace
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token, tokenizer.pad_token_id = (
        tokenizer.eos_token,
        tokenizer.eos_token_id,
    )

    # Sample input text
    prefix_text = "My name is Thomas and my main"

    # Initialize pipeline
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Run inference on Tenstorrent device
    answer = text_generator(
        prefix_text,
        max_length=30,
        num_beams=4,
        num_return_sequences=4,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    print(f"Prefix text: {prefix_text}")
    print("Generated text:")
    for sequence in answer:
        print(sequence.values())


if __name__ == "__main__":
    run_gpt2_text_gen()
