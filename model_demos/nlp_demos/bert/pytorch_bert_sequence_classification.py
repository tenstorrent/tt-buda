# BERT Demo Script - SST-2 Text Classification

import pybuda
from transformers import BertForSequenceClassification, BertTokenizer


def run_bert_sequence_classification_pytorch():

    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = "textattack/bert-base-uncased-SST-2"
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    model = BertForSequenceClassification.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_bert_sequence_classification", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1).item()

    # Answer - "positive"
    print(f"Review: {review} | Predicted Sentiment: {model.config.id2label[predicted_value]}")


if __name__ == "__main__":
    run_bert_sequence_classification_pytorch()
