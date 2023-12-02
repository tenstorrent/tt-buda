# DPR Demo Script - Reader

import pybuda
from transformers import DPRReader, DPRReaderTokenizer


def run_dpr_reader_pytorch(variant="facebook/dpr-reader-multiset-base"):

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = DPRReaderTokenizer.from_pretrained(model_ckpt)
    model = DPRReader.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Data preprocessing
    input_tokens = tokenizer(
        questions=["What is love?"],
        titles=["Haddaway"],
        texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_dpr_reader", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Postprocessing
    start_logits = output[0].value()
    end_logits = output[1].value()
    relevance_logits = output[2].value()

    # Print outputs
    print(f"Start Logits: {start_logits}")
    print(f"End Logits: {end_logits}")
    print(f"Relevance Logits: {relevance_logits}")


if __name__ == "__main__":
    run_dpr_reader_pytorch()
