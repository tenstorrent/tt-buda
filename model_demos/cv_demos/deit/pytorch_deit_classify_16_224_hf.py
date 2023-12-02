# DeiT Demo

import pybuda
import requests
from PIL import Image
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, ViTForImageClassification


def run_deit_classify_224_hf_pytorch(variant="facebook/deit-base-patch16-224"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # Create PyBuda module from PyTorch model
    image_processor = AutoFeatureExtractor.from_pretrained(variant)
    if variant == "facebook/deit-base-distilled-patch16-224":
        model = DeiTForImageClassificationWithTeacher.from_pretrained(variant)
    else:
        model = ViTForImageClassification.from_pretrained(variant)
    tt_model = pybuda.PyTorchModule("pt_deit_classif_16_224", model)

    # Load sample image
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
    print("Predicted class:", model.config.id2label[predicted_class_idx])


if __name__ == "__main__":
    run_deit_classify_224_hf_pytorch()
