# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest


import tensorflow as tf
import pybuda
from pybuda import (
    TFModule,
    VerifyConfig,
    CompileDepth,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module

def test_inceptionv3_tf(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    else:
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    input_shape = (1, 229, 229, 3)
    model = tf.keras.applications.InceptionV3(include_top=False, input_shape=input_shape[1:])
    mod = TFModule("inceptionv3_tf", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
