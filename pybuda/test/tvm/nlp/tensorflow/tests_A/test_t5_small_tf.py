# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from pybuda.module import TFModule
import pytest

import tensorflow as tf
import numpy as np
from transformers.models.t5.modeling_tf_t5 import TFT5Block, T5Config, TFT5Model
from test.backend.models.test_bert import get_relaxed_atol_pcc



from pybuda import (
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model
from loguru import logger


def test_t5_small(test_kind, test_device):
    pytest.skip()

    compiler_cfg = _get_global_compiler_config()

    pretrained_name = "t5-small"
    config = download_model(T5Config.from_pretrained, pretrained_name, torchscript=True)


    model = TFT5Model(config).get_decoder().block[0]
    mod = TFModule("t5_small_block_tf", model)
    input_shape = (1, 1, 128, 512)

    atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            atol=atol,
            pcc=pcc
        )
    )


def test_t5_small_fallback(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class T5Wrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def call(self, input_ids, decoder_input_ids):
            return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop=True 

    pretrained_name = "t5-small"
    config = download_model(T5Config.from_pretrained, pretrained_name)

    config.use_cache = False
    model = download_model(TFT5Model.from_pretrained, pretrained_name, config=config)
    mod = TFModule("t5_small", T5Wrapper(model))

    input_shape = (1, 128)
    verify_module(
        mod,
        (input_shape, input_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        input_params=[
            {"requires_grad": False, "data_format": tf.int32}, 
            {"requires_grad": False, "data_format": tf.int32},
        ],
    )
