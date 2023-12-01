# SqueezeBERT Demo Script - Text Classification

import pybuda
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run_squeezebert_sequence_classification_pytorch():

    # Load Bart tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("squeezebert/squeezebert-mnli")

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Example from multi-nli validation set
    text = """Hello, my dog is cute"""

    # Data preprocessing
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(pybuda.PyTorchModule("pt_bart", model), inputs=[(input_tokens,)])
    output = output_q.get()
    logits = output[0].value()

    # Data postprocessing
    scores = logits.softmax(dim=1)
    print("LABEL: SCORE")
    for i in range(len(model.config.id2label)):
        print(f"{model.config.id2label[i]}: {float(scores[0][i])}")


if __name__ == "__main__":
    run_squeezebert_sequence_classification_pytorch()
