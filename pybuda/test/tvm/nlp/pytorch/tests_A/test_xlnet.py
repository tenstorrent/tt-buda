# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from pybuda.config import CompileDepth
import pytest

import torch
from transformers import XLNetConfig, XLNetModel

from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    tvm_to_python,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

input_shapes = [(1, 16, 1024)]

@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
#TODO: Rewrite test for verify_module flow
def test_tvm_xlnet(training, recompute, input_shape):
    if training and not recompute:
        pytest.skip()
    
    compile_depth = CompileDepth.PRE_LOWERING_PASS
    if training:
        compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    config = XLNetConfig()

    model = XLNetModel(config)
    mod = PyTorchModule("XLNet", model.layer[0])

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    pos_emb = model.relative_positional_encoding(input_shape[0], input_shape[0], bsz=input_shape[1])
    target_mapping = torch.ones((1, 1, 16))
    output_g = model.mask_emb.expand(target_mapping.shape[0], 1, -1)

    attn_mask = torch.zeros((1, 16, 1, 1))
    non_tgt_mask = -torch.eye(16).to(attn_mask)
    non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)

    token_type_ids = torch.zeros((16, 1), dtype=torch.long)
    cat_ids = token_type_ids
    seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
    seg_mat = torch.nn.functional.one_hot(seg_mat, num_classes=2).to(torch.float)

    hidden_states = [torch.rand(*input_shape), output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat]

    ret = pybuda_compile(
        tt0,
        "XLNet",
        *hidden_states,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(model.layer[0], ret, *hidden_states)
