# OPT Demo Script - Question Answering

import os

import pybuda
from transformers import AutoTokenizer, OPTForQuestionAnswering


def run_opt_question_answering(variant="facebook/opt-350m"):

    # set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_ckpt = variant

    if model_ckpt == "facebook/opt-1.3b" or model_ckpt == "facebook/opt-350m":
        compiler_cfg.enable_auto_fusing = False
        if model_ckpt == "facebook/opt-1.3b":
            compiler_cfg.default_df_override = pybuda.DataFormat.Float16
            os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = OPTForQuestionAnswering.from_pretrained(model_ckpt, torchscript=True)

    # Load data sample
    question, context = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_opt_question_answering", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    answer_start = output[0].value().argmax().item()
    answer_end = output[1].value().argmax().item()
    answer = tokenizer.decode(input_tokens["input_ids"][0, answer_start : answer_end + 1])

    # Answer - "nice puppet"
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    run_opt_question_answering()
