# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import tensorflow as tf
from transformers import TFRobertaModel

from pybuda import (
    TFModule,
    TTDevice,
    BackendType,
    VerifyConfig,
    CPUDevice,
)
import pybuda

from pybuda.verify import verify_module, verify_module_pipeline
from pybuda.verify.config import TestKind

from pybuda.op.eval.common import compare_tensor_to_golden
from test.backend.models.test_bert import get_relaxed_atol_pcc
from test.utils import download_model


def test_roberta_encoder(test_kind, test_device):
    tf.keras.backend.clear_session()
    roberta_model = download_model(TFRobertaModel.from_pretrained, "arampacha/roberta-tiny", from_pt=True)
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    #if test_kind.is_training():
    #    test_device.devtype = BackendType.NoBackend

    class TF_RobertaEncoder(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.layer = model.layers[0].encoder

        def call(self, hidden_states):
            return self.layer(hidden_states, None, [None]*4, None, None, None, None, False, False, False)

    model = TF_RobertaEncoder(roberta_model)
    mod = TFModule("roberta_encoder_tf", model)

    input_shape = (1, 256, 256)
    hidden_states = tf.random.uniform(input_shape)

    waive_gradients = {"/attention/self/key/bias:0", "/attention/self/key/kernel:0", "/attention/self/query/bias:0", "/attention/self/query/kernel:0"}

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors=waive_gradients,
            relative_atol=relative_atol,
            pcc=pcc,
        )
    )
    

def test_roberta_pipeline(test_device, test_kind):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()
    tf.keras.backend.clear_session()

    roberta_model = download_model(TFRobertaModel.from_pretrained, "arampacha/roberta-tiny", from_pt=True)
    class EncoderWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.layers[0].encoder

        def call(self, hidden_states):
            return self.encoder(hidden_states, None, [None]*4, None, None, None, None, False, False, False)

    class EmbeddingsWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.embeddings = model.layers[0].embeddings

        def call(self, input_ids):
            return self.embeddings(input_ids)

    embeddings = TFModule("embeddings", EmbeddingsWrapper(roberta_model))
    encoder = TFModule("encoders", EncoderWrapper(roberta_model))

    microbatch_size = 1
    seq_len = 128
    vocab_size = roberta_model.layers[0].embeddings.config.vocab_size
    input_ids = tf.Variable(tf.random.uniform((microbatch_size, seq_len), maxval=vocab_size, dtype=tf.dtypes.int32), trainable=False)

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    waive_gradients = [f"roberta/encoder/layer_._{n}/attention/self/key/bias:0" for n in range(4)]
    verify_module_pipeline([embeddings, encoder],
            [(microbatch_size, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol, pcc=pcc,
                waive_gradient_errors=waive_gradients),
            inputs=[(input_ids, ), ],
            input_params=[{"requires_grad": False}],
            device_types=["CPUDevice", "TTDevice"],
    )

def test_roberta(test_device):
    tf.keras.backend.clear_session()
    roberta_model = download_model(TFRobertaModel.from_pretrained, "arampacha/roberta-tiny", from_pt=True)
    class EncoderWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.layers[0].encoder

        def call(self, hidden_states):
            return self.encoder(hidden_states, None, [None]*4, None, None, None, None, False, False, False)

    class EmbeddingsWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.embeddings = model.layers[0].embeddings

        def call(self, input_ids):
            return self.embeddings(input_ids)

    embeddings = EmbeddingsWrapper(roberta_model)
    encoder = EncoderWrapper(roberta_model)

    cpu0 = CPUDevice("cpu0", module=TFModule("embeddings", embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=TFModule("encoder", encoder))

    seq_len = 128
    vocab_size = roberta_model.layers[0].embeddings.config.vocab_size

    input_ids = tf.random.uniform((1, seq_len), maxval=vocab_size, dtype=tf.dtypes.int32)

    cpu0.push_to_inputs(tf.Variable(input_ids, trainable=False))
    output_q = pybuda.run_inference()
    outputs = output_q.get()

    tf_outputs = roberta_model(input_ids)
    torch_outputs = pybuda.tensor.to_pt_tensors(tf_outputs[0])
    
    assert compare_tensor_to_golden("roberta", torch_outputs[0], outputs[0].value(), is_buda=True)

