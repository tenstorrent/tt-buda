# DistilBERT Demo Script - SST-2 Text Classification

import pybuda
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


def run_distilbert_sequence_classification_pytorch():

    # Load DistilBert tokenizer and model from HuggingFace
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
    model = DistilBertForSequenceClassification.from_pretrained(model_ckpt)

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
        pybuda.PyTorchModule("pt_distilbert_sequence_classification", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1).item()

    # Answer - "positive"
    print(f"Review: {review} | Predicted Sentiment: {model.config.id2label[predicted_value]}")


if __name__ == "__main__":
    run_distilbert_sequence_classification_pytorch()
