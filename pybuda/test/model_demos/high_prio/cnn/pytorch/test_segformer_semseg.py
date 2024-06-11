import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

import os
import requests
import pytest
from PIL import Image


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


variants_semseg = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.parametrize("variant", variants_semseg)
def test_segformer_semseg_pytorch(test_device, variant):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    pcc_value = 0.99

    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        if variant in [
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-ade-512-512",
        ]:

            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        if (
            variant
            in [
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "nvidia/segformer-b2-finetuned-ade-512-512",
            ]
            and test_device.devtype == pybuda.BackendType.Silicon
        ):
            pcc_value = 0.98

    elif test_device.arch == pybuda.BackendDevice.Grayskull:

        if variant == "nvidia/segformer-b2-finetuned-ade-512-512":
            compiler_cfg.place_on_new_epoch("concatenate_1098.dc.concatenate.0")

        if variant == "nvidia/segformer-b3-finetuned-ade-512-512":
            compiler_cfg.place_on_new_epoch("concatenate_1890.dc.concatenate.0")

        if variant == "nvidia/segformer-b4-finetuned-ade-512-512":
            compiler_cfg.place_on_new_epoch("concatenate_2748.dc.concatenate.0")

        if test_device.devtype == pybuda.BackendType.Silicon:
            pcc_value = 0.98

    # Load the model from HuggingFace
    model = SegformerForSemanticSegmentation.from_pretrained(variant)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(pixel_values.shape,)],
        inputs=[(pixel_values,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_pybuda_codegen_vs_framework=True,
            verify_tvm_compile=True,
            pcc=pcc_value,
        ),
    )
