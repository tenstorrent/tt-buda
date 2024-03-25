# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import pybuda
import torch
import torch.nn as nn
import os
from pytorchcv.model_provider import get_model as ptcv_get_model
from transformers import BertModel, GPT2LMHeadModel, GPT2Config, GPT2Model, AutoFeatureExtractor, ResNetForImageClassification
from pybuda.torch_compile import compile_torch

def test_unet_osmr_cityscape_pytorch():
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.cpu_fallback_ops = set()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_FORCE_RESIZE_DENSE_MM"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"
    #if test_device.arch == BackendDevice.Wormhole_B0:
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
    #elif test_device.arch == BackendDevice.Grayskull:
    #    compiler_cfg.balancer_policy = "CNN"

    # STEP 2: Create PyBuda module from PyTorch model
    unet_osmr = ptcv_get_model("unet_cityscapes", pretrained=False)
    unet_osmr.eval()

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = torch.randn(1, 3, 224, 224)

    # Run the model on cpu
    golden = unet_osmr(img_tensor)

    # Run the model on TT device
    unet_osmr.to("tt")
    img_tensor = img_tensor.to("tt")
    pybuda_mod = torch.compile(unet_osmr, backend=compile_torch, dynamic=False)
    result = pybuda_mod(img_tensor)
    output = result[0].to("cpu")

    # Compare the result
    assert pybuda.op.eval.compare_tensor_to_golden(f"pt_unet_osmr_cityscape", golden[0], output, is_buda=True, pcc=0.99)


def test_resnet(): 
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops = set()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_training = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for output queue (perf)
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50", torchscript=True)
    resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", torchscript=True)
    resnet.eval()
 
    # Load data sample
    # url = "https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/train/18/image/image.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = torch.rand(1, 3, 256, 256)

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    
    # Run the model on cpu
    resnet_cpu = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", torchscript=True)
    golden = resnet_cpu(pixel_values)

    # Run the model on TT device
    resnet.to("tt")
    pixel_values = pixel_values.to("tt") 
    pybuda_mod = torch.compile(resnet, backend=compile_torch, dynamic=False)
    result = pybuda_mod(pixel_values)
    output = result[0].to("cpu")
    
    # Compare the result
    assert pybuda.op.eval.compare_tensor_to_golden(f"pt_resnet50", golden[0], output, is_buda=True, pcc=0.99)

def test_gpt2():
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 2

    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    compile_cfg.enable_link_past_cache_ios = True
    compile_cfg.cpu_fallback_ops = set()
    compile_cfg.default_df_override = pybuda._C.Float16_b

    gpt2 = GPT2LMHeadModel(config).eval()
    input_ids = torch.randint(0, 10000, (1, 32)).int()
    golden = gpt2(input_ids)

    pybuda_mod = torch.compile(gpt2, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)

    next_token_logits = result[0]
    next_token_logits = next_token_logits.to("cpu")

    res = result[0].to("cpu")
    assert pybuda.op.eval.compare_tensor_to_golden(f"gpt2", golden[0], res, is_buda=True, pcc=0.99)
    
def test_gen():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 1
    config.return_dict = False

    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    compile_cfg.enable_link_past_cache_ios = True
    compile_cfg.cpu_fallback_ops = set()
    compile_cfg.default_df_override = pybuda._C.Float16_b

    gpt2 = GPT2Model(config).eval()
    gpt2.to("tt")

    input_ids = torch.randint(0, 10000, (1, 32)).int().to("tt")

    pybuda_mod = torch.compile(gpt2, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)

    res = result[0].to("cpu")
    inp2 = torch.randint(0, 10000, (1, 32)).int()
    inp2 = inp2.to("tt")
    result = pybuda_mod(inp2, result[1])
    rs2 = result[0].to("cpu")

def test_bert():
    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    compile_cfg.cpu_fallback_ops = set()

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    bert_cpu = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)


    input_ids = torch.randint(0, 10000, (1, 128)).int()
    golden = bert_cpu(input_ids)

    print("Copying model")
    bert.to("tt")

    print("Copying inputs")
    input_ids = input_ids.to("tt")

    print("Compiling Model")
    pybuda_mod = torch.compile(bert, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)
    print("Copying outputs")

    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp2 = torch.randint(0, 10000, (1, 128)).int()
    golden = bert_cpu(inp2)

    inp2 = inp2.to("tt")
    result = pybuda_mod(inp2)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp3 = torch.randint(0, 10000, (1, 64)).int()
    golden = bert_cpu(inp3)
    inp3 = inp3.to("tt")
    result = pybuda_mod(inp3)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp4 = torch.randint(0, 10000, (1, 128)).int()
    golden = bert_cpu(inp4)
    inp4 = inp4.to("tt")
    result = pybuda_mod(inp4)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

    inp5 = torch.randint(0, 10000, (1, 64)).int()
    golden = bert_cpu(inp5)
    inp5 = inp5.to("tt")
    result = pybuda_mod(inp5)
    result = [r.to("cpu") for r in result]
    for i, (g, r) in enumerate(zip(golden, result)):
        assert pybuda.op.eval.compare_tensor_to_golden(f"bert_{i}", g, r, is_buda=True, pcc=0.99)

from diffusers import StableDiffusionPipeline

def test_sd():
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    prompt = "a photo of an astronaut riding a horse on mars"
    model = model.to("tt")
    pybuda_mod = torch.compile(model, backend=compile_torch)
    image = pybuda_mod(prompt=prompt, num_images_per_prompt=1, output_type="pil").images[0]
