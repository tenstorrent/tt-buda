# GPT Neo Demo Script - Sequence Classification

import os

import pybuda
from pybuda._C.backend_api import BackendDevice
from transformers import AutoTokenizer, GPTNeoForSequenceClassification


def run_gptneo_sequence_classification(variant="EleutherAI/gpt-neo-125M"):

    # Load tokenizer and model from HuggingFace
    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B
    model_ckpt = variant

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if "1.3B" in model_ckpt or "2.7B" in model_ckpt:
        os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

    if variant == "EleutherAI/gpt-neo-2.7B":
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForSequenceClassification.from_pretrained(model_ckpt, torchscript=True)

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_gptneo_seq_classification", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1).item()

    # Answer - "positive"
    print(f"Review: {review} | Predicted Sentiment: {model.config.id2label[predicted_value]}")


if __name__ == "__main__":
    run_gptneo_sequence_classification()
