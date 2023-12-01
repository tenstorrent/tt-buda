# RoBERTa demo script - Masked language modeling

import pybuda
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def run_roberta_mlm_pytorch():

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Input processing
    text = "Hello I'm a <mask> model."
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = torch.zeros_like(input_tokens)
    attention_mask[input_tokens != 1] = 1

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_roberta", model),
        inputs=[(input_tokens, attention_mask)],
    )
    output = output_q.get()  # inference will return a queue object, get last returned object

    # Output processing
    output_pt = output[0].value()
    scores = output_pt.softmax(dim=-1)
    mask_token_index = (input_tokens == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_rankings = output_pt[0, mask_token_index].argsort(axis=-1, descending=True)[0]

    # Report output
    top_k = 5
    print(f"Masked text: {text}")
    print(f"Top {top_k} predictions:")
    for i in range(top_k):
        prediction = tokenizer.decode(predicted_token_rankings[i])
        score = scores[0, mask_token_index, predicted_token_rankings[i]]
        print(f"{i+1}: {prediction} (score = {round(float(score), 3)})")


if __name__ == "__main__":
    run_roberta_mlm_pytorch()
