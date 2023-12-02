import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import _expand_mask, _make_causal_mask


class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, pixel_values):

        vision_outputs = self.clip_model.vision_model(pixel_values, return_dict=False)
        return vision_outputs


class CLIPTextWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, input_ids, attention_mask):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.clip_model.text_model.embeddings(input_ids=input_ids, position_ids=None)

        bsz, seq_len = input_shape
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.clip_model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            return_dict=False,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip_model.text_model.final_layer_norm(last_hidden_state)

        return (last_hidden_state, *encoder_outputs)


class CLIPPostProcessingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, input_ids, vision_outputs, last_hidden_state, *encoder_outputs):
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        text_outputs = (last_hidden_state, pooled_output) + encoder_outputs[1:]

        image_embeds = vision_outputs[1]
        image_embeds = self.clip_model.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.clip_model.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        output = (
            logits_per_image,
            logits_per_text,
            text_embeds,
            image_embeds,
            *text_outputs,
            *vision_outputs,
        )
        return output


def download_model(download_func, *args, num_retries=3, timeout=120, **kwargs):
    import shutil
    import time
    import urllib

    from loguru import logger

    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (requests.exceptions.HTTPError, urllib.error.HTTPError):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torch/models", ignore_errors=True)
            shutil.rmtree(
                os.path.expanduser("~") + "/.torchxrayvision/models_data",
                ignore_errors=True,
            )
            os.mkdir(os.path.expanduser("~") + "/.cache")
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."


def run_clip_pytorch(variant="openai/clip-vit-base-patch32"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    # Load processor and model from HuggingFace
    model_ckpt = variant
    model = download_model(CLIPModel.from_pretrained, model_ckpt, torchscript=True)
    processor = download_model(CLIPProcessor.from_pretrained, model_ckpt)

    # Load image from the IAM dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Process image
    text = [
        "a photo of a cat",
        "a photo of a dog",
    ]
    inputs = processor(text=text, images=image, return_tensors="pt")

    inputs = [
        inputs["input_ids"],
        inputs["pixel_values"],
        inputs["attention_mask"],
    ]
    vision_model = CLIPVisionWrapper(model)
    text_model = CLIPTextWrapper(model)
    post_processing_model = CLIPPostProcessingWrapper(model)

    vision_outputs = vision_model(inputs[1])

    tt0 = pybuda.TTDevice("tt0", module=pybuda.PyTorchModule("pt_clip_text_model", text_model))
    tt0.push_to_inputs(inputs[0], inputs[2])
    output_q = pybuda.run_inference(_sequential=True)
    text_outputs = output_q.get()
    text_outputs = [o.value().float() for o in text_outputs]

    post_processed_outputs = post_processing_model(inputs[0], vision_outputs, *text_outputs)
    probs = post_processed_outputs[0].softmax(dim=1)

    processed_output = list(zip(text, probs[0].tolist()))
    print("RESULTS")
    for item in processed_output:
        print(f"{item[0]}: {item[1]*100:.1f}%")
    print()


if __name__ == "__main__":
    run_clip_pytorch()
