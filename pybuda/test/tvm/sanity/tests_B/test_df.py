# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test data format control
"""
import pytest

import torch
import tensorflow as tf

import pybuda
import pybuda.op
from pybuda import (
    PyTorchModule,
    VerifyConfig,
    BackendDevice
)
from pybuda.verify import verify_module

verify_cfg = VerifyConfig(
    run_golden=True
)  # Run backend golden check on all tests in here


# PT 2.0 does not support float16 on cpu, so for now only run bfloat16, see tenstorrent/pybuda#1935
input_formats = [torch.bfloat16]
input_format_ids = ["bfloat16"]
weight_formats = [torch.bfloat16]
weight_format_ids = ["bfloat16"]


@pytest.mark.parametrize("input1_df", input_formats, ids=input_format_ids)
@pytest.mark.parametrize("input2_df", input_formats, ids=input_format_ids)
@pytest.mark.parametrize("weight_df", weight_formats, ids=weight_format_ids)
@pytest.mark.parametrize("force_matmul_spill", [False, True])
def test_pt_data_formats(test_kind, test_device, input1_df, input2_df, weight_df, force_matmul_spill):

    if input1_df != weight_df:
        pytest.skip()

    if test_device.arch == BackendDevice.Grayskull and input1_df != input2_df:
        pytest.skip() # For GS, no format-conversion allowed when matmul writes intermediates - BBE#1437

    class PyTorchTest(torch.nn.Module):
        """
        Simple torch module for data format testing
        """

        shape = (64, 64)

        def __init__(self, weight_type: torch.dtype):
            super().__init__()
            self.weights1 = torch.nn.Parameter(
                torch.rand(*self.shape, dtype=weight_type)
            )

        def forward(self, act1, act2):
            m1 = torch.matmul(act1, self.weights1)

            return m1 + act2

    framework_module = PyTorchTest(weight_df)
    module = PyTorchModule("pytorch_data_format", framework_module)

    if force_matmul_spill:
        pybuda.config.override_u_kt("matmul_1", 1)

    original_model_param_dtype = {}
    for key, val in framework_module.state_dict().items():
        original_model_param_dtype[key] = val.dtype

    verify_module(
        module,
        [
            (1, *PyTorchTest.shape),
            (1, *PyTorchTest.shape),
        ],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[
            {"data_format": input1_df},
            {"data_format": input2_df},
        ],
    )

    for key, val in module.module.state_dict().items():
        if original_model_param_dtype[key] != val.dtype:
            msg = "Original PyTorch model params have been modified (not allowed during compilation)."
            msg += f"Parameter '{key}', has dtype '{val.dtype}', while original is '{original_model_param_dtype[key]}'"
            pytest.fail(msg)


input_formats = [tf.float16, tf.bfloat16]
input_format_ids = ["float16", "bfloat16"]
weight_formats = [tf.float16, tf.bfloat16]
weight_format_ids = ["float16", "bfloat16"]


@pytest.mark.parametrize("input1_df", input_formats, ids=input_format_ids)
@pytest.mark.parametrize("input2_df", input_formats, ids=input_format_ids)
@pytest.mark.parametrize("weight_df", weight_formats, ids=weight_format_ids)
def test_tf_data_formats(test_kind, test_device, input1_df, input2_df, weight_df):

    if input1_df != weight_df:
        pytest.skip()

    if test_device.arch == BackendDevice.Grayskull and input1_df != input2_df:
        pytest.skip() # For GS, no format-conversion allowed when matmul writes intermediates - BBE#1437

    class TFTest(tf.keras.Model):
        """
        Simple tensorflow module for data format testing
        """

        shape = (64, 64)

        def __init__(self, weight_type: tf.dtypes.DType):
            super().__init__()
            self.weights1 = tf.Variable(
                tf.random.uniform(self.shape, dtype=weight_type)
            )

        def call(self, act1, act2):
            weights = self.weights1
            if weights.dtype != act1.dtype:
                weights = tf.cast(self.weights1, act1.dtype)
            m1 = tf.matmul(act1, weights)

            return m1 + act2

    framework_module = TFTest(weight_df)
    module = pybuda.TFModule("tensorflow_test", framework_module)

    original_model_param_dtype = []
    for val in framework_module.trainable_variables:
        original_model_param_dtype.append(val.dtype)

    verify_module(
        module,
        [
            (1, *TFTest.shape),
            (1, *TFTest.shape),
        ],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[
            {"data_format": input1_df},
            {"data_format": input2_df},
        ],
    )

    for i, val in enumerate(module.module.trainable_variables):
        if original_model_param_dtype[i] != val.dtype:
            msg = "Original TensorFlow model params have been modified (not allowed during compilation)."
            msg += f"Parameter '{val.name}', has dtype '{val.dtype}', while original is '{original_model_param_dtype[i]}'"
            pytest.fail(msg)
