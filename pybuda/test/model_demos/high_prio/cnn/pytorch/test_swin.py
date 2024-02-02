# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import PyBuda library
import pybuda

from transformers import ViTImageProcessor, SwinForImageClassification
import timm
import torch
import os

from test.utils import download_model
from datasets import load_dataset
from PIL import Image
import requests
import pytest

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


def test_swin_v1_tiny_4_224_hf_pytorch(test_device):
    pytest.skip() # Working on it
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()    
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.enable_tvm_constant_prop = True
    os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
    os.environ["TVM_BACKTRACE"]="1" 
    
    # STEP 2: Create PyBuda module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    # model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", torchscript=True)
    model = download_model(timm.create_model, "swin_tiny_patch4_window7_224", pretrained=True)
    tt_model = pybuda.PyTorchModule("Swin_v1_tiny_4_224", model)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    print(img_tensor.shape)
    # from pthflops import count_ops
    # flops = count_ops(model, img_tensor)
    #output = model(img_tensor).logits 
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()[0].value().detach().float().numpy()  
        
    predicted_class_idx = output.argmax(-1).item()
    print("Predicted class:", predicted_class_idx)
    print(model.config.id2label[predicted_class_idx])
