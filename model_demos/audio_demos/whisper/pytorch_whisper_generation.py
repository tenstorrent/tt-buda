# Whisper Demo - Conditional Generation

import os

import pybuda
from datasets import load_dataset
from pybuda._C.backend_api import BackendDevice
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def run_whisper_generation(variant="openai/whisper-small"):

    # PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.amp_level = 2
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"
    if "small" in variant:
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "35000"
    elif "medium" in variant or "large" in variant:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
    available_devices = pybuda.detect_available_devices()
    if available_devices[0] == BackendDevice.Grayskull:
        softmax_ops_to_override = [
            57,
            143,
            229,
            315,
            401,
            487,
            573,
            659,
            745,
            831,
            917,
            1003,
            1089,
            1175,
            1261,
            1347,
            1433,
            1519,
            1615,
            1691,
            1777,
            1863,
            1949,
            2035,
            1605,
            2121,
            2207,
            2293,
            2379,
            2465,
            2551,
            2637,
            2723,
        ]
        for op_id in softmax_ops_to_override:
            pybuda.config.override_op_size(f"softmax_{op_id}.dc.exp.2", (4, 1))
            pybuda.config.override_t_stream_shape(f"softmax_{op_id}.dc.exp.2", (1, 47))

    # Load processor and model from HuggingFace
    # Variants: openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large-v2
    model_ckpt = variant
    model = WhisperForConditionalGeneration.from_pretrained(model_ckpt)
    processor = WhisperProcessor.from_pretrained(model_ckpt)

    # Load sample from datasets
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_audio = ds[0]["audio"]["array"]

    # Initialize automatic-speech-recognition generator
    asr_pipeline = pybuda_pipeline(
        pipeline_type="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    # Run inference on Tenstorrent device
    answer = asr_pipeline(sample_audio)

    # Report output
    print("Generated text:", answer["text"])


if __name__ == "__main__":
    run_whisper_generation()
