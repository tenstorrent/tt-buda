# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np

import pybuda
import pybuda.op
from pybuda.op.loss import CrossEntropyLoss

from pybuda import (
    PyBudaModule,
    TTDevice,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
)
from pybuda._C.backend_api import BackendType
from loguru import logger


TEST_VOCAB_SIZE = 128
TEST_VOCAB_SAMPLING_RANGE = 128
TEST_SEQUENCE_LENGTH = 128


def cross_entropy_loss_torch_ops(predictions, labels):
    exponential_predicted = torch.exp(predictions)
    reduced_exponential_predicted = torch.sum(
        exponential_predicted, dim=1, keepdim=True
    )
    ln_reduced_exponential_predicted = torch.log(
        reduced_exponential_predicted
    )
    output_minus_ln_reduced_exponential_predicted = (
        predictions - ln_reduced_exponential_predicted
    )
    negative_one_times_output_minus_ln_reduced_exponential_predicted = (
        -1.0 * output_minus_ln_reduced_exponential_predicted
    )
    word_losses = (
        negative_one_times_output_minus_ln_reduced_exponential_predicted * labels
    )
    token_losses = torch.sum(word_losses, dim=1)
    avg_sequence_loss = torch.mean(token_losses, dim=0)
    return avg_sequence_loss


def test_bert_cross_entropy_loss_torch():

    predictions_shape = (1, 1, TEST_SEQUENCE_LENGTH, TEST_VOCAB_SIZE)
    mlm_labels_shape = (1, 1, TEST_SEQUENCE_LENGTH, TEST_VOCAB_SIZE)

    predictions = torch.rand(*predictions_shape)
    pytorch_masked_lm_labels = torch.empty(128, dtype=torch.long).random_(
        TEST_VOCAB_SAMPLING_RANGE
    )
    buda_masked_lm_labels = torch.zeros(*mlm_labels_shape)

    for i, class_label in enumerate(pytorch_masked_lm_labels.view(-1)):
        buda_masked_lm_labels[:, :, i, class_label] = 1.0

    pytorch_loss = torch.nn.CrossEntropyLoss()(
        predictions.view(-1, TEST_VOCAB_SIZE), pytorch_masked_lm_labels.view(-1)
    )
    pytorch_ops_loss = cross_entropy_loss_torch_ops(
        predictions.view(-1, TEST_VOCAB_SIZE),
        buda_masked_lm_labels.view(TEST_SEQUENCE_LENGTH, TEST_VOCAB_SIZE),
    )

    assert torch.allclose(pytorch_loss, pytorch_ops_loss)


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_bert_cross_entropy_loss_pybuda(training, recompute):
    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    mod = CrossEntropyLoss("cross_entropy_loss")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    predictions_shape = (1, 1, TEST_SEQUENCE_LENGTH, TEST_VOCAB_SIZE)
    mlm_labels_shape = (1, 1, TEST_SEQUENCE_LENGTH, TEST_VOCAB_SIZE)

    predictions = torch.rand(predictions_shape, requires_grad=True)
    pytorch_masked_lm_labels = torch.empty(128, dtype=torch.long).random_(
        TEST_VOCAB_SAMPLING_RANGE
    )
    buda_masked_lm_labels = torch.zeros(mlm_labels_shape)

    for i, class_label in enumerate(pytorch_masked_lm_labels.view(-1)):
        buda_masked_lm_labels[0, 0, i, class_label] = 1.0

    predictions_tensor = Tensor.create_from_torch(predictions)
    buda_masked_lm_labels_tensor = Tensor.create_from_torch(buda_masked_lm_labels)

    ret = pybuda_compile(
        tt0,
        "cross_entropy_loss",
        predictions_tensor,
        buda_masked_lm_labels_tensor,
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(),
    )
    pybuda_loss = torch.sum(ret.golden_outputs[0]) # torch.sum to reduce from tile-size 32,32
    pytorch_loss = torch.nn.CrossEntropyLoss()(
        predictions.view(-1, TEST_VOCAB_SIZE), pytorch_masked_lm_labels.view(-1)
    )
    torch.allclose(pybuda_loss, pytorch_loss)


class PyBudaTwoOutputModule(PyBudaModule):
    """
    dummy module to mimic conditions like the pretraining head with two outputs
    we can then feed into the loss module
    """
    shape = (1, 1, TEST_SEQUENCE_LENGTH, TEST_VOCAB_SIZE)
    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        def mprefix(layer_name: str) -> str:
            # Prepends name with module-name prefix
            return self.get_name() + "." + layer_name

        m1 = pybuda.op.Matmul(mprefix("matmul1"), act1, self.weights1)
        m2 = pybuda.op.Matmul(mprefix("matmul2"), act2, self.weights2)
        return m1, m2


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_bert_fwd_and_loss_module(training, recompute):
    if not training and recompute:
        pytest.skip() # inference + recompute is the same as just inference

    forward_module = PyBudaTwoOutputModule("fwd")
    loss_module = CrossEntropyLoss("cross_entropy_loss")

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(forward_module)
    tt0.place_module(loss_module)

    act1 = Tensor.create_from_torch(torch.rand(*PyBudaTwoOutputModule.shape))
    act2 = Tensor.create_from_torch(torch.rand(*PyBudaTwoOutputModule.shape, requires_grad=True))

    # TODO(jchu): verify -> probably better to commonize under the Tensor api, modify do_verify(..) since
    # it currently expects loss as a pytorch tensor
    if training:
        #SINGLE_TILE_SHAPE = (1, 1, 32, 32)
        initial_loss = torch.ones((1, 1, 1, 1), requires_grad=False)
        losses = (initial_loss, )
    else:
        losses = None

    forward_module.set_parameter("weights1", torch.rand(*PyBudaTwoOutputModule.shape, requires_grad=True))
    forward_module.set_parameter("weights2", torch.rand(*PyBudaTwoOutputModule.shape, requires_grad=True))

    pybuda_compile(
        tt0,
        "sanity",
        act1,
        act2,
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(),
        losses=losses
    )
