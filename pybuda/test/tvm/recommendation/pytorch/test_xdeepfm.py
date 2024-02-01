# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
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
from pybuda.config import CompileDepth
from test.tvm.utils import evaluate_framework_vs_pybuda
import pytest
from deepctr_torch.models.xdeepfm import xDeepFM

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

from test.tvm.recommendation.pytorch.deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, xDeepFM, WDL

import os

from pybuda.config import CompileDepth, _get_global_compiler_config

key2index = {}

def split(x):
    key_ans = x.split("|")
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


def test_xdeepfm_cin(test_kind, test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "deepctr_torch/movielens_sample.txt"))
    sparse_features = [
        "movie_id",
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip",
    ]

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    genres_list = list(map(split, data["genres"].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding="post",)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature


    fixlen_feature_columns = [
        SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
        for feat in sparse_features
    ]

    varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat("genres", vocabulary_size=len(key2index) + 1, embedding_dim=4),
            maxlen=max_len,
            combiner="mean",
            length_name=None,
        )
    ]  # Notice : value 0 is for padding for sequence input feature


    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    model = xDeepFM(
        linear_feature_columns,
        dnn_feature_columns,
        task="regression",
    )

    mod = PyTorchModule("xdeepfm_cin", model.cin)

    input_shape = (1, 7, 100)

    verify_module(
        mod,
        (input_shape, ),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_xdeepfm_dnn(test_kind, test_device):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "deepctr_torch/movielens_sample.txt"))
    sparse_features = [
        "movie_id",
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip",
    ]

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    genres_list = list(map(split, data["genres"].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding="post",)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature


    fixlen_feature_columns = [
        SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
        for feat in sparse_features
    ]

    varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat("genres", vocabulary_size=len(key2index) + 1, embedding_dim=4),
            maxlen=max_len,
            combiner="mean",
            length_name=None,
        )
    ]  # Notice : value 0 is for padding for sequence input feature


    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    model = xDeepFM(
        linear_feature_columns,
        dnn_feature_columns,
        task="regression",
    )
    mod = PyTorchModule("xdeepfm_dnn", model.dnn)

    input_shape = (1, 28)

    verify_module(
        mod,
        (input_shape, ),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )
