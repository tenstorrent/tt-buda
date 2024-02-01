# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Falcon-7B Demo Script

import pytest
from pybuda import BackendDevice, BackendType
from test.model_demos.models.falcon.model import Falcon

def test_falcon_pytorch(test_device):
    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    if test_device.devtype == BackendType.Golden:
        pytest.skip()

    # Load model from HuggingFace``
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