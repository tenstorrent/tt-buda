# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Longformer basic bring-up tests of tracing functionality
#
import pytest

import torch
from transformers import LongformerModel


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


def test_longformer(training):
    pytest.skip()  # Due to the current issues with unknown attention shapes during compile
    # time (possible attention concatenation issues), we'll pausing further support for the 
    # Longformer model for now. 
    #
    # The main problem is related to the nonzero op which has unknown shape during compile
    # time. Nonzero op is mostly used to find the indexes of global attention indices in the
    # phase of calculating the global attention probabilities.

    recompute = True  # Always run with recompute in post-commit CI. Nightly tests both

    if not training:
        compile_depth = CompileDepth.FULL
    else:
        compile_depth = CompileDepth.FULL

    model = LongformerModel.from_pretrained(
        "allenai/longformer-base-4096", torchscript=True
    )

    submodel = model
    module = PyTorchModule(
        "longformer",
        submodel,
    )

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    input_shape = (1, 4096)
    input_ids = torch.randint(0, 9000, input_shape, dtype=torch.long)
    attention_mask = torch.ones(
        input_ids.shape, dtype=torch.long, device=input_ids.device
    )
    global_attention_mask = torch.zeros(
        input_ids.shape, dtype=torch.long, device=input_ids.device
    )
    global_attention_mask[
        :, [1, 4, 21]
    ] = 1  # Randomly set global attentions for testing purposes

    res = pybuda_compile(
        tt0,
        "longformer",
        input_ids,
        attention_mask,
        global_attention_mask,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(
        submodel, res, input_ids, attention_mask, global_attention_mask
    )


def test_longformer_layer(training):
    pytest.skip() # Due to the current issues with unknown attention shapes during compile
    # time (possible attention concatenation issues), we'll pausing further support for the 
    # Longformer model for now. 
    #
    # The main problem is related to the nonzero op which has unknown shape during compile
    # time. Nonzero op is mostly used to find the indexes of global attention indices in the
    # phase of calculating the global attention probabilities.

    recompute = True  # Always run with recompute in post-commit CI. Nightly tests both

    if not training:
        compile_depth = CompileDepth.FULL
    else:
        compile_depth = CompileDepth.FULL

    model = LongformerModel.from_pretrained(
        "allenai/longformer-base-4096", torchscript=True
    )

    submodel = model.encoder.layer[0]
    module = PyTorchModule(
        "longformer",
        submodel,
    )

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    input_act_shape = (1, 512, 768)
    attention_mask_shape = (1, 512)
    layer_head_mask_shape = (model.config.num_attention_heads)

    act = torch.rand(input_act_shape)
    attention_mask = torch.zeros(attention_mask_shape, dtype=torch.long)
    attention_mask[0][0] = 1
    attention_mask[0][1] = -1
    attention_mask[0][2] = 1
    attention_mask[0][3] = 1
    attention_mask[0][-1] = 1
    layer_head_mask = torch.rand(layer_head_mask_shape)
    is_index_masked = attention_mask < 0
    is_index_global_attn = attention_mask > 0
    is_global_attn = is_index_global_attn.flatten().any()

    inputs = [
        act,
        attention_mask,
        layer_head_mask,
        is_index_masked,
        is_index_global_attn,
        is_global_attn
    ]

    res = pybuda_compile(
        tt0,
        "longformer",
        *inputs,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(submodel, res, inputs)
