# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind, DataFormat

import os

import pybuda
import requests
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


dataset = load_dataset("huggingface/cats-image")
image_1 = dataset["test"]["image"][0]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_2 = Image.open(requests.get(url, stream=True).raw)


def generate_model_vit_imgcls_hf_pytorch(test_device, variant):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # STEP 2: Create PyBuda module from PyTorch model
    image_processor = download_model(AutoImageProcessor.from_pretrained,
        variant
    )
    model = download_model(ViTForImageClassification.from_pretrained,
        variant
    )
    tt_model = pybuda.PyTorchModule("ViT_classif_16_224", model)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits
    
    return tt_model, [img_tensor], {}


variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classify_224_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device, variant,
    )

    if "PYBUDA_NEB_GALAXY_CI" in os.environ:
        chip_ids = [0, 11, 10, 9, 8, 7, 19, 20, 21, 22, 23, 24, 6, 5, 14, 13, 12, 16, 15, 3, 4, 26, 25, 32, 31, 30, 29, 28, 27, 1, 2, 18, 17]
    else:
        chip_ids = [0]

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
            chip_ids=chip_ids
        )
    )

variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant, already tested with test_vit_classification_1x1_demo")
def test_vit_classify_224_hf_pytorch_1x1(variant, test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    if "large" in variant:
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "20000"


    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device, variant,
    )

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9
        )
    )

modes = [
    "verify",
    "demo"
]
variants = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
]
@pytest.mark.parametrize("mode", modes, ids=modes)
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classification_1x1_demo(test_device, mode, variant):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip("Not supported")

    # Setup for 1x1 grid
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    
    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON2"] = "1"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.enable_tvm_cpu_fallback = False
    
    # Load image preprocessor and model
    image_processor = download_model(AutoImageProcessor.from_pretrained,  variant)
    framework_model = download_model(ViTForImageClassification.from_pretrained, variant)
    model_name = "_".join(variant.split('/')[-1].split('-')[:2]) + f"_{mode}"
    tt_model = pybuda.PyTorchModule(model_name, framework_model)

    # Load and preprocess image
    dataset = load_dataset("huggingface/cats-image")
    input_image = dataset["test"]["image"][0]
    input_image = image_processor(input_image, return_tensors="pt").pixel_values
    
    if mode == "verify":
        # Verify model on Tenstorrent device
        verify_module(
            tt_model,
            input_shapes=[(input_image.shape,)],
            inputs=[(input_image,)],
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                devmode=test_device.devmode,
                test_kind=TestKind.INFERENCE,
            )
        )
    elif mode == "demo":
        # Run inference on Tenstorrent device
        output_q = pybuda.run_inference(tt_model, inputs=([input_image]))
        output = output_q.get()[0].value().detach().float().numpy()

        # Postprocessing
        predicted_class_idx = output.argmax(-1).item()

        # Print output
        print("Predicted class:", framework_model.config.id2label[predicted_class_idx])
