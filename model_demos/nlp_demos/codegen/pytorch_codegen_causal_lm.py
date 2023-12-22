# CodeGen Demo - CasualLM
# Support for Wormhole only

import os

import pybuda
from pybuda._C.backend_api import BackendDevice
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, CodeGenForCausalLM


def run_codegen_causal_lm(variant="Salesforce/codegen-350M-mono"):

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{32*1024}"
    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.default_dram_parameters = False
            compiler_cfg.balancer_policy = "Ribbon"
    # DRAM stream limit
    compiler_cfg.balancer_op_override("matmul_1829", "grid_shape", (2, 8))

    # Load tokenizer and model
    # Variants: Salesforce/codegen-350M-mono, Salesforce/codegen-350M-multi, Salesforce/codegen-350M-nl
    # Salesforce/codegen-2B-mono, Salesforce/codegen-2B-multi, Salesforce/codegen-2B-nl
    model_ckpt = variant
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = CodeGenForCausalLM.from_pretrained(model_ckpt, use_cache=False)

    # Set special PAD token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Set prompt
    prompt = "def hello_world():"

    # Run inference on Tenstorrent device
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    answer = text_generator(
        prompt,
        max_length=20,
        num_beams=1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    print(f"Prompt: {prompt}")
    print("Generated text:")
    for sequence in answer:
        print(sequence.values())


if __name__ == "__main__":
    run_codegen_causal_lm()
