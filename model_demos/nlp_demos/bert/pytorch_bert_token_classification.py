# BERT Demo Script - NER

import pybuda
import torch
from transformers import BertForTokenClassification, BertTokenizer


def run_bert_token_classification_pytorch():

    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    model = BertForTokenClassification.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = "HuggingFace is a company based in Paris and New York"

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
        pybuda.PyTorchModule("pt_bert_token_classification", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    predicted_token_class_ids = output[0].value()[0].argmax(-1)
    predicted_token_class_ids = torch.masked_select(predicted_token_class_ids, (input_tokens["attention_mask"][0] == 1))
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]

    # Answer - ['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']
    print(f"Context: {sample_text}")
    print(f"Answer: {predicted_tokens_classes}")


if __name__ == "__main__":
    run_bert_token_classification_pytorch()
