# BeiT Model Demo

import os

import pybuda
import requests
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from transformers import BeitForImageClassification, BeitImageProcessor


def run_beit_classify_224_hf_pytorch(variant="microsoft/beit-base-patch16-224"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    available_devices = pybuda.detect_available_devices()

    compiler_cfg.enable_t_streaming = True
    if variant == "microsoft/beit-base-patch16-224":
        compiler_cfg.retain_tvm_python_files = True
        compiler_cfg.enable_tvm_constant_prop = True
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
    elif variant == "microsoft/beit-large-patch16-224":
        if available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.retain_tvm_python_files = True
            compiler_cfg.enable_tvm_constant_prop = True
            os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
        else:
            compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    image_processor = BeitImageProcessor.from_pretrained(variant)
    model = BeitForImageClassification.from_pretrained(variant)
    tt_model = pybuda.PyTorchModule("pt_beit_classif_16_224", model)

    # Get sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    img_tensor = image_processor(sample_image, return_tensors="pt").pixel_values

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get()[0].value().detach().float().numpy()

    # Postprocessing
    predicted_class_idx = output.argmax(-1).item()

    # Print output
    print("Predicted class:", predicted_class_idx)
    print(model.config.id2label[predicted_class_idx])


if __name__ == "__main__":
    run_beit_classify_224_hf_pytorch()
