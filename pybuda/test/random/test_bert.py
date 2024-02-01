# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from pybuda.verify import verify_module, VerifyConfig, TestKind

from test.bert.modules import (
    PyBudaBertEncoder,
    get_bert_parameters
)
    
def test_encoder(test_index, random_seeds, test_device):
    hidden_dim = 384
    num_heads = 6
    seq_len = 384

    microbatch_size = 8

    params = get_bert_parameters("encoder", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertEncoder("encoder", params, config)
    params["reciprocal_of_sqrt_of_head_size_0"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(num_heads)))

    # Ignore pcc errors, we don't care about them here - with random formats and AMP, it's not going to be particularly accurate
    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch, pcc=0.1),
            input_params=[{}, {"requires_grad": False}],
    )

