# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest 

import os
import onnx
import tensorflow as tf
import pybuda
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda.verify.config import TestKind

def test_lstm_valence_onnx(test_device):
    # Load model checkpoint from HuggingFace
    load_path = "third_party/confidential_customer_models/model_2/onnx/lstm_valence/lstm-valence-model.onnx"
    model = onnx.load(load_path)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    
    # Required to patch data-mismatch. Here is followup issue
    # to check this out in more details:
    # tenstorrent/pybuda#1828
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"

    # Run inference on Tenstorrent device
    inputs = tf.random.uniform(shape=[1, 1, 282])   
    verify_module(
        pybuda.OnnxModule("onnx_lstm", model, load_path),
        input_shapes=(inputs.shape,),
        inputs=[(inputs,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.98,
        )
    )
