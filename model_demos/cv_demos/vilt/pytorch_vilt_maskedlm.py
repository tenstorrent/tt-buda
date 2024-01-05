import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import ViltConfig, ViltForMaskedLM, ViltProcessor

from .vilt_model import ViLtEmbeddingWrapper, ViltModelWrapper


def run_vilt_maskedlm_pytorch(variant="dandelin/vilt-b32-mlm"):

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
    sample_text = "a bunch of cats laying on a [MASK]."

    model_ckpt = variant

    # Set model configurations
    config = ViltConfig.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = ViltProcessor.from_pretrained(model_ckpt)
    model = ViltForMaskedLM.from_pretrained(model_ckpt, config=config)
    model.eval()

    # prepare inputs
    encoding = processor(sample_image, sample_text, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model=model, task="maskedlm", text_seq_len=encoding["input_ids"].shape[1])

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    tt0 = pybuda.TTDevice("tt0", module=pybuda.PyTorchModule("pt_vilt_maskedlm", vilt_model))
    tt0.push_to_inputs((embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)))

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(_sequential=True)
    mlm_logits = output_q.get()[0].value().detach().float()

    # PostProcessing
    input_ids = encoding["input_ids"][0][1:-1]
    mlm_logits = mlm_logits[0, 1 : encoding.input_ids.shape[1] - 1, :]

    mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
    mlm_values[input_ids != 103] = 0
    select = mlm_values.argmax().item()
    inferred_token = processor.decode(mlm_ids[select].item())

    # Model Output (i.e Masked token: Couch)
    print("Masked token: ", inferred_token)


if __name__ == "__main__":
    run_vilt_maskedlm_pytorch()
