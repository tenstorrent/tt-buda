import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

from .vilt_model import ViLtEmbeddingWrapper, ViltModelWrapper


def run_vilt_for_question_answering_pytorch(variant="dandelin/vilt-b32-finetuned-vqa"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Sample Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_image = Image.open(requests.get(url, stream=True).raw)

    # Sample text
    sample_text = "How many cats are there?"

    model_ckpt = variant

    # Set model configurations
    config = ViltConfig.from_pretrained(model_ckpt)  # matmul_2008
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = ViltProcessor.from_pretrained(model_ckpt)
    model = ViltForQuestionAnswering.from_pretrained(model_ckpt, config=config)
    model.eval()

    # Sample inputs
    encoding = processor(sample_image, sample_text, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    viltquestionanswering_model = ViltModelWrapper(model, task="qa")

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    tt0 = pybuda.TTDevice("tt0", module=pybuda.PyTorchModule("pt_viltquestionanswering", viltquestionanswering_model))

    tt0.push_to_inputs(embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32))

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(_sequential=True)

    # Model output (i.e Predicted answer: 2)
    output = output_q.get()[0].value().detach().float()
    idx = output.argmax(-1).item()
    print("Predicted answer: ", model.config.id2label[idx])


if __name__ == "__main__":
    run_vilt_for_question_answering_pytorch()
