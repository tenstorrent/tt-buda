# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Various non-real unit testing models for performance sanity and debug
"""

import pybuda
import torch

from ..common import benchmark_model

# NLP feedforward bock
class NLPFF(pybuda.PyBudaModule):
    def __init__(self, name: str, c: int):
        super().__init__(name)
        self.ff1_w = pybuda.Parameter(torch.rand(c, c * 4), requires_grad=True)
        self.ff1_bias = pybuda.Parameter(torch.rand(1, c*4), requires_grad=True)
        self.ff2_w = pybuda.Parameter(torch.rand(c * 4, c), requires_grad=True)
        self.ff2_bias = pybuda.Parameter(torch.rand(1, c), requires_grad=True)

    def forward(self, act):
        act = pybuda.op.Matmul("ff1", act, self.ff1_w)
        act = pybuda.op.Add("ff1_bias", act, self.ff1_bias)
        act = pybuda.op.Gelu("gelu", act)
        act = pybuda.op.Matmul("ff2", act, self.ff2_w)
        act = pybuda.op.Add("ff2_bias", act, self.ff2_bias)
        return act

# NLP self-attention batched matmul
class BMM(pybuda.PyBudaModule):
    def __init__(self, name: str):
        super().__init__(name)
        self.bias = pybuda.Parameter(torch.rand(1, 1024), requires_grad=True)

    def forward(self, k, q, v):
        k = pybuda.op.HSlice("k_slice", k, 16)
        q = pybuda.op.HSlice("q_slice", q, 16)
        v = pybuda.op.HSlice("v_slice", v, 16)
        qt = pybuda.op.Transpose("qt", q, -1, -2)
        out = pybuda.op.Matmul("bmm0", pybuda.op.Matmul("bmm1", k, qt), v)
        out = pybuda.op.HStack("hstack", out)
        return out + self.bias


@benchmark_model(configs=["nlp-ff-base", "nlp-ff-large", "nlp-bmm"])
def unit(training: bool, config: str, microbatch: int, devtype: str, arch: str):

    if microbatch == 0:
        microbatch = 64

    targets = []
    if config == "nlp-ff-base":
        models = {"tt": NLPFF(config, 768)}
        inputs = [torch.rand(microbatch, 128, 768)]
        if training:
            targets = [torch.rand(microbatch, 128, 768)]
    elif config == "nlp-ff-large":
        models = {"tt": NLPFF(config, 1024)}
        inputs = [torch.rand(microbatch, 384, 1024)]
        if training:
            targets = [torch.rand(microbatch, 384, 1024)]

    elif config == "nlp-bmm":
        models = {"tt": BMM("bmm")}
        inputs = [torch.rand(microbatch, 384, 1024), torch.rand(microbatch, 384, 1024), torch.rand(microbatch, 384, 1024)] 
        if training:
            targets = [torch.rand(microbatch, 384, 1024)]

    else:
        assert(False)

    if training:
        models["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return models, inputs, targets, {}

