# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import pybuda
import torch
import torch.nn as nn
import os
import requests
from PIL import Image
from datasets import load_dataset
from pytorchcv.model_provider import get_model as ptcv_get_model
from transformers import BertModel, GPT2LMHeadModel, GPT2Config, GPT2Model, AutoFeatureExtractor, ResNetForImageClassification
from pybuda.torch_compile import compile_torch
from typing import Tuple


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
    pytest.skip()   # Working on it
    config = GPT2Config.from_pretrained("gpt2")
    config.num_hidden_layers = 1

    os.environ["PYBUDA_DEVMODE"] = "1"
    compile_cfg = pybuda.config._get_global_compiler_config()
    # compile_cfg.enable_link_past_cache_ios = True
    compile_cfg.cpu_fallback_ops = set()
    compile_cfg.default_df_override = pybuda._C.Float16_b

    gpt2 = GPT2LMHeadModel(config).eval()
    gpt2.to("tt")

    input_ids = torch.randint(0, 10000, (1, 32)).int().to("tt")
    # past_cache_shape = (1, 12, 96, 64)
    # past_cache = []
    # for _ in range(config.num_hidden_layers):
    #     past_cache.append((torch.zeros(past_cache_shape).to("tt"), torch.zeros(past_cache_shape).to("tt")))
    # past_cache = tuple(past_cache)

    pybuda_mod = torch.compile(gpt2, backend=compile_torch, dynamic=False)
    result = pybuda_mod(input_ids)

    res = result[0].to("cpu")
    breakpoint()
    inp2 = torch.randint(0, 10000, (1, 32)).int()
    inp2 = inp2.to("tt")
    result = pybuda_mod(inp2, result[1])
    
def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return x1 + x2, x2 + x1 + 2

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = Add()
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    golden = model(*inputs)
    pybuda_mod = torch.compile(model, backend=compile_torch)
    # inputs = [i.to("tt") for i in inputs]
    result = pybuda_mod(*inputs)
    result = [r.to("cpu") for r in result]

    assert [torch.allclose(g, r) for g, r in zip(golden, result)]

def test_conv2d():
    class Conv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = Conv2d()
    inputs = torch.rand(1, 3, 32, 32)
    golden = model(inputs) 

    if True:
        pybuda_mod = torch.compile(model, backend=compile_torch, dynamic=False)
        result = pybuda_mod(inputs)
        result = result.to("cpu")
        assert pybuda.op.eval.compare_tensor_to_golden(f"conv2d", golden, result, is_buda=True, pcc=0.99)
    else: 
        from pybuda.verify.backend import verify_module
        mod = pybuda.PyTorchModule("conv", model)
        verify_module(
            mod,
            ([1,3,32,32],),
            verify_cfg=pybuda.VerifyConfig(
                arch=pybuda.BackendDevice.Wormhole_B0,
                devtype=pybuda.BackendType.Golden,
                test_kind=pybuda.verify.TestKind.INFERENCE,
                pcc=0.99
            ), 
        )

def test_bn():
    class BN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x):
            x = self.bn(x)
            return x

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = BN()
    model.eval()

    inputs = torch.rand(1, 64, 32, 32)
    golden = model(inputs)
    # inputs = [i.to("tt") for i in inputs]
    pybuda_mod = torch.compile(model, backend=compile_torch)
    result = pybuda_mod(inputs)
    result = result.to("cpu")

    assert pybuda.op.eval.compare_tensor_to_golden(f"linear", golden, result, is_buda=True, pcc=0.99)

def test_linear():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 64, bias=True)

        def forward(self, x1, x2):
            m1 = self.linear(x1)
            return m1 + x2

    os.environ["PYBUDA_DEVMODE"] = "1"
    model = Linear()
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 64)]
    golden = model(*inputs)
    # inputs = [i.to("tt") for i in inputs]
    pybuda_mod = torch.compile(model, backend=compile_torch)
    result = pybuda_mod(*inputs)
    result = result.to("cpu")

    assert pybuda.op.eval.compare_tensor_to_golden(f"linear", golden, result, is_buda=True, pcc=0.99)

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


from torch._dynamo import export
from torch._decomp import register_decomposition
import torch
import torch.nn as nn

torch._dynamo.reset()
import torch._dynamo as dynamo


def test_decomp():
    pytest.skip() #TODO fix: FATAL    | Always          - Unsupported (for now) _copy_from TTDevice[0] to TTDevice[0]
    os.environ["PYBUDA_DEVMODE"] = "1"
    class BasicModule(nn.Module):
        def forward(self, x):
            x = x * 2
            a,b,c = torch.split(x, 3, dim=-1)
            return a + b + c

    mod, input = BasicModule(), torch.randn(2, 9).to(dtype=torch.float16)
 
    pybuda_mod = torch.compile(mod, backend=compile_torch, dynamic=False)
    out = pybuda_mod(input)

@pytest.mark.parametrize("shape", [(1024, 1024)])
@pytest.mark.parametrize("mb", [1, 8, 16])
@pytest.mark.parametrize("loop", [1, 8, 16])
@pytest.mark.parametrize("native", [True, False])
def test_push(shape, mb, loop, native):
    if mb != 1:
        pytest.skip() #TODO
    os.environ["PYBUDA_DEVMODE"] = "1"
    import time

    pybuda.config.set_configuration_options(
        default_df_override=pybuda.config.DataFormat.Float32
    )

    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return x1 + x2

    model = Add()
    sample_inputs = [torch.empty(mb, 1, *shape), torch.empty(mb, 1, *shape)]
    inputs = [(torch.ones(mb, 1, *shape), torch.ones(mb, 1, *shape))] * loop

    if native:
        model = model.to("tt")
        pybuda_mod = pybuda_mod = torch.compile(model, backend=compile_torch, dynamic=False)
        comp_inputs = [i.to("tt") for i in inputs[0]]
        result = pybuda_mod(*comp_inputs) # compile
        start = time.perf_counter()
        for args in inputs:
            args = [a.to("tt") for a in args]
            result = pybuda_mod(*args)
            result.to("cpu")
        elapsed = time.perf_counter() - start
    else:
        tt0 = pybuda.TTDevice("tt0")
        tt0.place_module(pybuda.module.PyTorchModule("add", model))
        output_q = pybuda.initialize_pipeline(
            training=False, sample_inputs=sample_inputs
        )

        start = time.perf_counter()
        for i in range(loop):
            tt0.push_to_inputs(inputs[i])
        pybuda.run_forward(input_count=loop)
        for i in range(loop):
            result = output_q.get(timeout=30)
        elapsed = time.perf_counter() - start

    float32_size = 4
    data = mb * shape[0] * shape[1] * float32_size / (1024 * 1024)

    print(
        f"Batch[{mb:2}] Loop[{loop:2}] Native[{native:1}] Data[{data}mB] Elapsed[{elapsed:2.4}sec]"
    )
