# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch
import pytest
from transformers import GemmaModel, GemmaConfig
from transformers import AutoTokenizer, GemmaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

import pybuda
from pybuda import (
    VerifyConfig,
    PyTorchModule,
    CompileDepth,
)
from test.utils import download_model
from pybuda.pybudaglobal import TILE_DIM
from pybuda.verify.config import TestKind
from pybuda._C import DataFormat, MathFidelity
from pybuda._C.backend_api import BackendDevice
from pybuda._C.backend_api import BackendType
from pybuda.verify.backend import verify_module
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline


def cpu_sanity_run_0():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


def cpu_sanity_run_1():
    model = GemmaForCausalLM.from_pretrained("google/gemma-2b")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(generated_text)


variants = [
    "google/gemma-2b",
]


@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_rotary_embedding(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    
    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].self_attn.rotary_emb

        def forward(self, x, pos_ids):
            cos, sin = self.model(x, pos_ids)
            
            return cos, sin

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    pytorch_model = Wrapper(pytorch_model)
    tt_model = PyTorchModule("pytorch_gemma_2b_rotary_embedding", pytorch_model)

    # Define inputs
    x = torch.rand((1, 1, 7, 256)).to(torch.float32)
    pos_ids = torch.arange(7).unsqueeze(0).to(torch.float32)
    
    # Sanity run
    out = pytorch_model(x, pos_ids)
    print(out)

    verify_module(
        tt_model,
        input_shapes=[(x.shape, pos_ids.shape,)],
        inputs=[(x, pos_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_rms_norm(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].input_layernorm

        def forward(self, x):
            out = self.model(x)
            
            return out

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    pytorch_model = Wrapper(pytorch_model)
    tt_model = PyTorchModule("pytorch_gemma_2b_rms_norm", pytorch_model)

    # Define inputs
    x = torch.rand((1, 7, 2048)).to(torch.float32)
    
    # Sanity run
    out = pytorch_model(x)
    print(out)

    verify_module(
        tt_model,
        input_shapes=[(x.shape,)],
        inputs=[(x,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_attention(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].self_attn

        def forward(self, hidden_states, attn_mask, pos_ids):
            attn_output, attn_weights, past_key_value = self.model(hidden_states, attn_mask, pos_ids)
            
            return attn_output

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    pytorch_model = Wrapper(pytorch_model)
    tt_model = PyTorchModule("pytorch_gemma_2b_attention", pytorch_model)

    # Define inputs
    hidden_states = torch.rand((1, 7, 2048)).to(torch.float32)
    attn_mask = torch.ones((1, 1, 7, 7)).to(torch.float32)
    pos_ids = torch.arange(7).unsqueeze(0).to(torch.float32)
    
    # Sanity run
    out = pytorch_model(hidden_states, attn_mask, pos_ids)
    print(out)

    verify_module(
        tt_model,
        input_shapes=[(hidden_states.shape, attn_mask.shape, pos_ids.shape,)],
        inputs=[(hidden_states, attn_mask, pos_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_mlp(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].mlp

        def forward(self, hidden_states):
            out = self.model(hidden_states)
            
            return out

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    pytorch_model = Wrapper(pytorch_model)
    tt_model = PyTorchModule("pytorch_gemma_2b_attention", pytorch_model)

    # Define inputs
    x = torch.rand((1, 7, 2048)).to(torch.float32)
    
    # Sanity run
    out = pytorch_model(x)
    print(out)

    verify_module(
        tt_model,
        input_shapes=[(x.shape,)],
        inputs=[(x,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_single_decoder(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0]

        def forward(self, hidden_states, attn_mask, pos_ids):
            out = self.model(hidden_states, attn_mask, pos_ids)
            
            return out

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    pytorch_model = Wrapper(pytorch_model)
    tt_model = PyTorchModule("pytorch_gemma_2b_single_decoder", pytorch_model)

    # Define inputs
    hidden_states = torch.rand((1, 7, 2048)).to(torch.float32)
    attn_mask = torch.ones((1, 1, 7, 7)).to(torch.float32)
    pos_ids = torch.arange(7).unsqueeze(0).to(torch.float32)
    
    # Sanity run
    out = pytorch_model(hidden_states, attn_mask, pos_ids)
    print(out)

    verify_module(
        tt_model,
        input_shapes=[(hidden_states.shape, attn_mask.shape, pos_ids.shape,)],
        inputs=[(hidden_states, attn_mask, pos_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.skip(reason="Tested as part of a full generative model run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    tt_model = PyTorchModule("pytorch_gemma_2b", pytorch_model)
    
    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=30)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"Sanity run generated text: {generated_text}")
    
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    
    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape, attn_mask.shape,)],
        inputs=[(input_ids, attn_mask,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.skip(reason="Tested as part of a full generative model run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_1x1(test_device, variant):
    pytest.xfail("Passing locally, failing on CI. Keeping as XFail to be able to track potential regressions.")
    
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    elif test_device.arch == BackendDevice.Blackhole:
        os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "blackhole_1x1.yaml"
    
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    tt_model = PyTorchModule("pytorch_gemma_2b_1x1", pytorch_model)
    
    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=30)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"Sanity run generated text: {generated_text}")
    
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    
    verify_module(
        tt_model,
        input_shapes=[(input_ids.shape, attn_mask.shape,)],
        inputs=[(input_ids, attn_mask,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_gen(test_device, variant):
    # Random seed for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    if test_device.arch != BackendDevice.Grayskull:
        compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
        
        # Configure all matmul ops to operate on HiFi4 with Bfp8_b inputs/params and Float16 accumulation
        pybuda.config.configure_mixed_precision(
            op_type='matmul',
            math_fidelity=MathFidelity.HiFi4,
            input_df={0:[DataFormat.Bfp8_b, False], 1:[DataFormat.Bfp8_b, False]},
            accumulate_df=DataFormat.Float16_b
        )

        # Configure all other ops to run on HiFi4 with Float16 accumulation
        pybuda.config.configure_mixed_precision(
            op_type='^((?!matmul).)*$',
            math_fidelity=MathFidelity.HiFi4,
            accumulate_df=DataFormat.Float16_b
        )
    
    if test_device.arch == BackendDevice.Grayskull:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{65*1024}"

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    
    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=30)
    generated_pt_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nPyTorch (sanity) generated:")
    pt_ans = generated_pt_text.split('\n\n')[1]
    print(f"{pt_ans}")
    
    # Initialize and Run text2text generator on Tenstorrent device
    text2text_generator = pybuda_pipeline(
        "text2text-generation",
        model=pytorch_model,
        tokenizer=tokenizer,
        pybuda_max_length=32,
    )
    generated_tt_text = text2text_generator(
        prompt,
        max_length=32,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    
    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nTT generated:")
    for sequence in generated_tt_text:
        tt_ans = sequence['generated_text'][len(prompt):]
        print(f"{tt_ans}")


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_1x1_gen(test_device, variant):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Not supporting the Grayskull 1x1 overlay yet")
    
    # Random seed for reproducibility
    torch.manual_seed(42)

    compiler_cfg = pybuda.config._get_global_compiler_config()
    
    if test_device.devtype == BackendType.Silicon and "CI_PROJECT_DIR" in os.environ: 
        pytest.skip("Failing on CI with Read 0xffffffff from ARC scratch[6]: you should reset the board")

    # Configurations
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    # Configure all matmul ops to operate on HiFi4 with Bfp8_b inputs/params and Float16 accumulation
    pybuda.config.configure_mixed_precision(
        op_type='matmul',
        math_fidelity=MathFidelity.HiFi4,
        input_df={0:[DataFormat.Bfp8_b, False], 1:[DataFormat.Bfp8_b, False]},
        accumulate_df=DataFormat.Float16_b
    )

    # Configure all other ops to run on HiFi4 with Float16 accumulation
    pybuda.config.configure_mixed_precision(
        op_type='^((?!matmul).)*$',
        math_fidelity=MathFidelity.HiFi4,
        accumulate_df=DataFormat.Float16_b
    )

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    
    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    
    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=30)
    generated_pt_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nPyTorch (sanity) generated:")
    pt_ans = generated_pt_text.split('\n\n')[1]
    print(f"{pt_ans}")
    
    # Initialize and Run text2text generator on Tenstorrent device
    text2text_generator = pybuda_pipeline(
        "text2text-generation",
        model=pytorch_model,
        tokenizer=tokenizer,
        pybuda_max_length=32,
    )
    generated_tt_text = text2text_generator(
        prompt,
        max_length=32,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    
    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nTT generated:")
    for sequence in generated_tt_text:
        tt_ans = sequence['generated_text'][len(prompt):]
        print(f"{tt_ans}")


if __name__ == "__main__":
    test_gemma_2b_gen()
