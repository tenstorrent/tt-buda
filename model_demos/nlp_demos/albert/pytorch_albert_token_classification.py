# ALBERT Demo Script - NER

import os

import pybuda
import torch
from pybuda._C.backend_api import BackendDevice
from transformers import AlbertForTokenClassification, AlbertTokenizer


def run_albert_token_classification_pytorch(size="base", variant="v2"):
    available_devices = pybuda.detect_available_devices()

    # Set PyBUDA configuration parameters
    pybuda.config.set_configuration_options(
        default_df_override=pybuda.DataFormat.Float16,
        amp_level=2,
    )

    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    # Variants: albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1
    # albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
    model_ckpt = f"albert-{size}-{variant}"
    if "xxlarge" in model_ckpt:
        pybuda.config.set_configuration_options(
            enable_t_streaming=True,
            enable_auto_fusing=False,
            enable_enumerate_u_kt=False,
        )
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{105*1024}"
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "xlarge" in model_ckpt:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{8*1024}"
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg = pybuda.config._get_global_compiler_config()
                compiler_cfg.enable_t_streaming = True
                os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "large" in model_ckpt:
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
                compiler_cfg = pybuda.config._get_global_compiler_config()
                compiler_cfg.enable_t_streaming = True
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"

    # Load ALBERT tokenizer and model from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    model = AlbertForTokenClassification.from_pretrained(model_ckpt)

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
        pybuda.PyTorchModule("pt_albert_token_classification", model),
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
    run_albert_token_classification_pytorch()
