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
)
from test.utils import download_model
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module


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


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

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


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_1x1(test_device, variant):
    # Random see for reproducibility
    torch.manual_seed(42)
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON2"] = "1"
    # compiler_cfg.amp_level = 1
    # os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "20000"
    # os.environ["PYBUDA_ENABLE_SINGLE_BUFFER_FALLBACK"] = "1"
    # compiler_cfg.enable_single_buffer_fallback = True
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


if __name__ == "__main__":
    test_gemma_2b()
