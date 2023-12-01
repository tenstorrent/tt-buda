# DPR Demo Script - Question Encoder

import pybuda
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


def run_dpr_question_encoder_pytorch(
    variant="facebook/dpr-question_encoder-multiset-base",
):

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-question_encoder-single-nq-base, facebook/dpr-question_encoder-multiset-base
    model_ckpt = variant
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_ckpt)
    model = DPRQuestionEncoder.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = "Hello, is my dog cute?"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_dpr_question_encoder", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Postprocessing
    embeddings = output[0].value()

    # Print embeddings
    print(f"Context: {sample_text}")
    print(f"Embeddings: {embeddings}")


if __name__ == "__main__":
    run_dpr_question_encoder_pytorch()
