# Falcon-7B Demo Script

import pybuda
import pytest
from pybuda._C.backend_api import BackendDevice

from nlp_demos.falcon.utils.model import Falcon


def run_falcon_pytorch():
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            pytest.skip()

    # Load model from HuggingFace
    model = Falcon(
        user_rows=32,  # Current limitation of model is that input must be batch 32
        num_tokens=128,
        top_p_enable=1,
        top_p=0.9,
        top_k=40,
        temperature=1.0,
        model_ckpt="tiiuae/falcon-7b-instruct",
    )
    model.initialize()

    # Load sample prompt
    sample_text = "What are the names of some famous actors that started their careers on Broadway?"

    # Generate batch of 32 prompts
    inputs = [sample_text] * 32

    # Run inference on Tenstorrent device
    outputs = model.inference(inputs)

    # Display output
    print(f"Answer: {outputs[0]}")


if __name__ == "__main__":
    run_falcon_pytorch()
