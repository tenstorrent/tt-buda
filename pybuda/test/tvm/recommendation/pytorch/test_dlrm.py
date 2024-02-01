# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2018, deepakn94, robieta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy
from pybuda.verify.backend import verify_module

import pytest
import pybuda

m_spa = (None,)
ln_emb = (None,)
ln_bot = (None,)
ln_top = (None,)

# NOTE: we are using default dlrm params to make a dummy network
default_params_toy = {
    "m_spa": 16,
    "ln_emb": numpy.array([4, 3, 2]),
    "ln_bot": numpy.array([13, 512, 256, 64, 16]),
    "ln_top": numpy.array([22, 512, 256, 1]),
    "arch_interaction_itself": False,
    "md_threshold": 200,
    "loss_threshold": 0.0,
    "qr_threshold": 200,
    "ndevices": -1,
    "sigmoid_bot": -1,
    "qr_operation": "mult",
    "qr_flag": False,
    "qr_collisions": 4,
    "sigmoid_top": 2,
    "md_flag": False,
    "arch_interaction_op": "dot",
    "sync_dense_params": True,
}
# NOTE: we are using default dlrm params to make a dummy network
default_params_small = {
    "m_spa": 8,
    "ln_emb": numpy.array([32, 16, 16]),
    "ln_bot": numpy.array([32, 16, 4]),
    "ln_top": numpy.array([25, 16, 9, 1]),
    "arch_interaction_itself": False,
    "md_threshold": 200,
    "loss_threshold": 0.0,
    "qr_threshold": 200,
    "ndevices": -1,
    "sigmoid_bot": -1,
    "qr_operation": "mult",
    "qr_flag": False,
    "qr_collisions": 0,
    "sigmoid_top": 2,
    "md_flag": False,
    "arch_interaction_op": "dot",
    "sync_dense_params": True,
}
# NOTE: we are using default dlrm params to make a dummy network
default_params_bench = {
    "m_spa": 128,
    "ln_emb": numpy.array([4, 3, 2]),
    "ln_bot": numpy.array([13, 512, 256, 128]),
    "ln_top": numpy.array([134, 1024, 1024, 512, 256, 1]),
    "arch_interaction_itself": False,
    "md_threshold": 200,
    "loss_threshold": 0.0,
    "qr_threshold": 200,
    "ndevices": -1,
    "sigmoid_bot": -1,
    "qr_operation": "mult",
    "qr_flag": False,
    "qr_collisions": 0,
    "sigmoid_top": 2,
    "md_flag": False,
    "arch_interaction_op": "dot",
    "sync_dense_params": True,
}

import numpy as np
import torch
import torch.nn as nn
import sys
from os.path import abspath, join, dirname

from pybuda.config import CompileDepth, _get_global_compiler_config

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            # FIXME: going with default embedding bag for toy model
            #
            #            if self.qr_flag and n > self.qr_threshold:
            #                EE = QREmbeddingBag(n, m, self.qr_collisions,
            #                    operation=self.qr_operation, mode="sum", sparse=True)
            #            elif self.md_flag and n > self.md_threshold:
            #                _m = m[i]
            #                base = max(m)
            #                EE = PrEmbeddingBag(n, _m, base)
            #                # use np initialization as below for consistency...
            #                W = np.random.uniform(
            #                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
            #                ).astype(np.float32)
            #                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            #

            # else:
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)

            emb_l.append(EE)

        return emb_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments

            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            # create operators
            if ndevices <= 1:
                self.emb_l = self.create_emb(m_spa, ln_emb)
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    # def apply_emb(self, lS_o, lS_i, emb_l):
    def apply_emb(self, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            # sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, torch.tensor([0]))

            ly.append(V)

        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])

            # FIXME: original tensor indexing causes broken graph
            # Zflat = Z[:, li, lj]

            # FIXME: This is a temperory fix to concatenate tensors in two steps
            data = [Z[:, i, j] for i, j in zip(li, lj)]
            Zflat = torch.cat(data).view(1, len(data))

            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)

        else:
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)

        return R

    # def forward(self, dense_x, lS_o, lS_i):
    def forward(self, dense_x, lS_i):
        dense_x = dense_x.squeeze(0)
        lS_i = [elem.squeeze(0) for elem in lS_i]
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)

        # process sparse features(using embeddings), resulting in a list of row vectors
        # ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        ly = self.apply_emb(lS_i, self.emb_l)

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z


class DLRM_interact_features(nn.Module):
    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
    ):
        super(DLRM_interact_features, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments

            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

    def forward(self, x, ly_0, ly_1, ly_2):
        ly = [ly_0, ly_1, ly_2]
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])

            # FIXME: original tensor indexing causes broken graph
            # Zflat = Z[:, li, lj]

            # FIXME: This is a temperory fix to concatenate tensors in two steps
            data = [Z[:, i, j] for i, j in zip(li, lj)]
            Zflat = torch.cat(data).view(1, len(data))

            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)

        else:
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)

        return R




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


def test_dlrm_mlp_bot(test_kind, test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.POST_PATTERN_MATCHER
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    dlrm = DLRM_Net(**default_params_toy)
    mod = PyTorchModule("dlrm_mlp_bot", dlrm.bot_l)

    input_shape = (1, default_params_toy["ln_bot"][0])

    verify_module(
        mod,
        (input_shape, ),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_dlrm_interact(test_kind, test_device):

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    dlrm = DLRM_interact_features(**default_params_toy)
    mod = PyTorchModule("dlrm_interact", dlrm)

    input_shapes = ((1, 16) for i in range(4))

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


@pytest.mark.parametrize("size", ["toy", "small", "bench"])
def test_dlrm(test_kind, test_device, size):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip()
    if test_kind.is_training():
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    if size == "toy":
        default_params = default_params_toy
    elif size == "small":
        default_params = default_params_small
    elif size == "bench":
        default_params = default_params_bench
    else:
        assert False

    dlrm = DLRM_Net(**default_params)
    mod = PyTorchModule("dlrm", dlrm)

    batch = 20
    dense_x = torch.tensor(
        np.random.random(size=(batch, 1, default_params["ln_bot"][0])).astype(np.float32)
    )
    lS_i = [torch.tensor([[1, 2, 3],]).to(torch.int).repeat(batch, 1), torch.tensor([[1, 2, 1],]).to(torch.int).repeat(batch, 1), torch.tensor([[0, 1, 0],]).to(torch.int).repeat(batch, 1)]

    verify_module(
        mod,
        None,
        inputs=[(dense_x, lS_i), ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
    )