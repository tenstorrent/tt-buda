# ALBERT Demo Script - SST-2 Text Classification

import pybuda
from transformers import AlbertForSequenceClassification, AlbertTokenizer


def run_albert_sequence_classification_pytorch():

    # Set PyBUDA configuration parameters
    pybuda.config.set_configuration_options(
        default_df_override=pybuda.DataFormat.Float16,
        amp_level=2,
    )

    # Load ALBERT tokenizer and model from HuggingFace
    model_ckpt = "textattack/albert-base-v2-imdb"
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    model = AlbertForSequenceClassification.from_pretrained(model_ckpt)

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
        pybuda.PyTorchModule("pt_albert_sequence_classification", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1).item()

    # Answer - "positive"
    print(f"Review: {review} | Predicted Sentiment: {model.config.id2label[predicted_value]}")


if __name__ == "__main__":
    run_albert_sequence_classification_pytorch()
