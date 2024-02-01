# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import tensorflow as tf
from transformers import DistilBertConfig
from transformers.models.distilbert.modeling_tf_distilbert import (
    TFDistilBertModel,
    TFTransformer,
)

from pybuda import (
    TFModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.backend.models.test_bert import get_relaxed_atol_pcc
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.utils import download_model


def test_distilbert_tf(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Transformer(tf.keras.Model):
        def __init__(self, module):
            super().__init__()
            self.attn_mask = tf.ones((1, 128))
            self.module = module

        def call(self, input_act):
            return self.module(input_act, self.attn_mask)

    # Load the model
    framework_module = download_model(
        TFDistilBertModel.from_pretrained, 
        "distilbert-base-cased-distilled-squad"
    )
    framework_module = Transformer(framework_module)
    pybuda_module = TFModule("distilbert_tf", framework_module)

    # Input shapes
    input_act_shape = (1, 128)

    # Sanity check
    # act = tf.random.uniform(input_act_shape, 0, 25000, dtype=tf.int32)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
            waive_gradient_errors={
                "attention/k_lin/bias:0",
            },
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )


def test_distilbert_layer_tf(test_kind, test_device):
    pytest.skip("Already tested in DistilBert full model")
    if test_kind == TestKind.TRAINING:  # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 32, 768)
    configuration = DistilBertConfig()

    class Transformer(tf.keras.Model):
        def __init__(self, config, attn_mask):
            super().__init__()
            self.config = config
            self.layer = TFTransformer(config)
            self.attn_mask = attn_mask

        def call(self, hidden_states):
            return self.layer(
                hidden_states,
                self.attn_mask,
                [None] * self.config.n_layers,
                False,
                False,
                False,
            )

    # Initializing a model from the configuration
    model = Transformer(configuration, tf.ones(input_shape[0:2]))
    submodel = model
    mod = TFModule("distilbert_transformer_tf", submodel)

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, "tiny", 1)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"attention/k_lin/bias"},
            relative_atol=relative_atol,
            pcc=pcc,
        ),
        uniform_inputs=True,
    )
