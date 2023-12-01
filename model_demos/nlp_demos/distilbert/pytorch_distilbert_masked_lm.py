# DistilBERT Demo Script - Masked LM

import pybuda
from transformers import DistilBertForMaskedLM, DistilBertTokenizer


def run_distilbert_masked_lm_pytorch(variant="distilbert-base-uncased"):

    # Load DistilBert tokenizer and model from HuggingFace
    # Variants: distilbert-base-uncased, distilbert-base-cased,
    # distilbert-base-multilingual-cased
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_ckpt = variant
    tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
    model = DistilBertForMaskedLM.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = "The capital of France is [MASK]."

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
        pybuda.PyTorchModule("pt_distilbert_masked_lm", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    mask_token_index = (input_tokens["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = output[0].value()[0, mask_token_index].argmax(axis=-1)
    answer = tokenizer.decode(predicted_token_id)

    # Answer - "paris"
    print(f"Context: {sample_text}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    run_distilbert_masked_lm_pytorch()
