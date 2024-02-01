# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from audioop import bias

from pybuda.verify.backend import verify_module
import torch
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from test.tvm.utils import evaluate_framework_vs_pybuda
import pytest

import torch

from deepctr_torch.models import AFN
from deepctr_torch.utils import get_test_data

@pytest.mark.parametrize(
    'afn_dnn_hidden_units, sparse_feature_num, dense_feature_num',
    [((256, 128), 3, 0),
     ((256, 128), 3, 3),
     ((256, 128), 0, 3)]
)
def test_afn_dnn(test_kind, test_device, afn_dnn_hidden_units, sparse_feature_num, dense_feature_num):

    _get_global_compiler_config().compile_depth = CompileDepth.POST_PATTERN_MATCHER
    sample_size = 64
    _, _, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFN(feature_columns, feature_columns, afn_dnn_hidden_units=afn_dnn_hidden_units)

    mod = PyTorchModule("afn_dnn", model.afn_dnn)

    verify_module(
        mod,
        ((32, model.embedding_size*256),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


@pytest.mark.parametrize(
    'afn_dnn_hidden_units, sparse_feature_num, dense_feature_num',
    [((256, 128), 3, 0),
     ((256, 128), 3, 3),
     ((256, 128), 0, 3)]
)
def test_afn_dnn_linear(test_kind, test_device, afn_dnn_hidden_units, sparse_feature_num, dense_feature_num):

    sample_size = 64
    _, _, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFN(feature_columns, feature_columns, afn_dnn_hidden_units=afn_dnn_hidden_units)

    mod = PyTorchModule("afn_dnn_linear", model.afn_dnn_linear)

    verify_module(
        mod,
        ((1, 32, 128),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )


@pytest.mark.parametrize(
    'afn_dnn_hidden_units, sparse_feature_num, dense_feature_num',
    [((256, 128), 3, 0)]
)
def test_afn_ltl(test_kind, test_device, afn_dnn_hidden_units, sparse_feature_num, dense_feature_num):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    if test_kind.is_training():
        # clip not defined in eltwise unary backward
        _get_global_compiler_config().compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    sample_size = 64
    _, _, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFN(feature_columns, feature_columns, afn_dnn_hidden_units=afn_dnn_hidden_units)

    mod = PyTorchModule("afn_ltl", model.ltl)

    verify_module(
        mod,
        ((1, 6, 4),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )

