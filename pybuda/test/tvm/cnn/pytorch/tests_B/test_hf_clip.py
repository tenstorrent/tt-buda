# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch


from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, CLIPConfig

import pybuda
from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind

class ClipWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, attention_mask):
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        return outputs.logits_per_image


def test_hf_clip(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    compiler_cfg.retain_tvm_python_files = True
    
    config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    config.text_config.num_hidden_layers=1
    config.vision_config.num_hidden_layers=1
    model = CLIPModel(config)

    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    pad_len = 32 - (inputs["input_ids"].shape[1] % 32)
    input_ids = torch.nn.functional.pad(inputs["input_ids"], (0, pad_len))
    attention_mask = torch.nn.functional.pad(inputs["attention_mask"], (0, pad_len))
    pixel_values = inputs["pixel_values"]

    verify_module(
        PyTorchModule("clip", ClipWrapper(model)),
        (input_ids.shape, pixel_values.shape, attention_mask.shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        inputs=[(input_ids, pixel_values, attention_mask)],
    )
    # tt0 = pybuda.TTDevice("tt0", 
    #         devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("clip", ClipWrapper(model)))
    # tt0.push_to_inputs((input_ids, pixel_values, attention_mask))
    # output_q = pybuda.run_inference()
    # outputs = output_q.get()

    # outputs = model(input_ids, pixel_values, attention_mask)
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
