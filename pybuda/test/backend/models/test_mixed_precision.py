# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import pybuda
from pybuda.verify import verify_module, VerifyConfig
from pybuda import DataFormat, PyTorchModule
from transformers import BertModel, BertConfig

def get_relaxed_atol_pcc(test_kind, test_device, size = "tiny", microbatch_size = 1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.95
    if test_device.is_silicon():
        training_pcc = 0.85
        if size != "tiny" or microbatch_size > 1:
            training_atol = 0.55
            training_pcc = 0.8
    inference_atol = 0.1
    inference_pcc = 0.95
    relative_atol = training_atol if test_kind.is_training() else inference_atol
    if test_device.is_silicon() and test_kind.is_training():
        relative_atol *= 3.5
    pcc = training_pcc if test_kind.is_training() else inference_pcc

    return relative_atol, pcc


@pytest.mark.parametrize("encoder_count", [1], ids=["enc1",])
def test_pt_encoder(test_kind, test_device, encoder_count):
    if test_kind.is_training():
        pytest.skip()

    # Set Mixed Precision Settings
    pybuda.config.configure_mixed_precision(
        op_type="softmax", output_df=DataFormat.Float16_b
    )

    # using tiny configuration
    size = "tiny"
    model_name = "prajjwal1/bert-tiny"
    seq_len = 128

    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = encoder_count # no need to run a lot
    model = BertModel(config=config)
    encoder = PyTorchModule("bert_encoder", model.encoder)
    microbatch = 1

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, size, microbatch)

    waive_gradient_errors = {"attention.self.key.bias"}
    verify_module(encoder, [(microbatch, seq_len, config.hidden_size), (microbatch, 1, seq_len, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc,
                optimizer=None,
                accumulation_steps=1,
                microbatch_count=1,
                epochs=1,
                chip_ids=list(range(1)),
                waive_gradient_errors=waive_gradient_errors),
            # Input gradient is really hard to match, so setting requires_grad to false.
            # Will need another way to say what's "correct"
            input_params=[{"requires_grad": False}, {"requires_grad": False}],
            uniform_inputs=True,
    )
