# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pybuda
from pybuda.cpudevice import CPUDevice
from pybuda.op.eval.common import compare_tensor_to_golden
from pybuda.verify.backend import verify_module
from test.tvm.recommendation.pytorch.deepctr_torch.inputs import SparseFeat, combined_dnn_input
from test.tvm.recommendation.pytorch.deepctr_torch.models import DeepFM
from test.tvm.recommendation.pytorch.deepctr_torch.utils import SAMPLE_SIZE, check_model, get_device, get_test_data
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
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

class DeepFMWrapper(DeepFM):

    def embed(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)
        return logit, *sparse_embedding_list, *dense_value_list

    def recommend(self, logit, sparse_embedding_list, dense_value_list):
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit = logit + self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit = logit + dnn_logit

        y_pred = self.out(logit)
        return y_pred

class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embeddings = model.embed

    def forward(self, X):
        return self.embeddings(X)

class RecommendationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.recommend = model.recommend
        self.num_embeddings = len(model.embedding_dict)

    def forward(self, logit, *embedding_list):
        sparse_embedding_list = embedding_list[:self.num_embeddings]
        dense_value_list = embedding_list[self.num_embeddings:]
        y_pred = self.recommend(logit, sparse_embedding_list, dense_value_list)
        return y_pred
    
@pytest.mark.skip(reason="Unsupported HW ops: concatenate")
@pytest.mark.parametrize(
    'use_fm,hidden_size,sparse_feature_num,dense_feature_num',
    [(True, (32,), 3, 3),
     (False, (32,), 3, 3),
     (False, (32,), 2, 2),
     (False, (32,), 1, 1),
     (True, (), 1, 1),
     (False, (), 2, 2),
     (True, (32,), 0, 3),
     (True, (32,), 3, 0),
     (False, (32,), 0, 3),
     (False, (32,), 3, 0),
     ]
)
def test_DeepFM(test_device, use_fm, hidden_size, sparse_feature_num, dense_feature_num):
    
    sample_size = 1
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    x = [x[feature] for feature, _ in x.items()]
    for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
    x = torch.from_numpy(np.concatenate(x, axis=-1).astype('float32'))

    model = DeepFMWrapper(feature_columns, feature_columns, use_fm=use_fm,
                   dnn_hidden_units=hidden_size, dnn_dropout=0.5)
    model.eval()

    relative_atol = 0.3 if test_device.is_silicon() else 0.1

    embeddings = EmbWrapper(model)
    recommendation = RecommendationWrapper(model)

    cpu0 = CPUDevice("cpu0", module=PyTorchModule("deepfm_embeddings", embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("deepfm_recommendation", recommendation))

    # x = torch.randn(input_shape)
    torch_outputs = model(x)
    cpu0.push_to_inputs(x)
    output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(relative_atol=relative_atol))
    outputs = output_q.get()

    
    assert compare_tensor_to_golden("deepfm", torch_outputs[0], outputs[0].value(), is_buda=True, relative_atol=relative_atol)


def test_deepfm_fm(test_kind, test_device):

    #Fusing disabled due to tenstorrent/pybuda#789
    pybuda.set_configuration_options(enable_auto_fusing=False)

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "deepctr_torch/movielens_sample.txt"))
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    device = 'cpu'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    mod = PyTorchModule("deepfm_fm", model.fm)

    input_shape = (1, 256, 1024)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )
