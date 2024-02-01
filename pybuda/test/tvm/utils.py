# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict

import torch
import tensorflow as tf
from loguru import logger

from pybuda.op.eval.common import compare_tensor_to_golden


def evaluate_framework_vs_pybuda(
    framework_model, pybuda_model_results, *inputs, rtol=None, atol=None
):
    """
    Evaluates PyTorch model results agains compiled PyBuda model results.

    Args:
        framework_model: Framework model.
        pybuda_model_results (CompileResults): PyBuda model results.
        rtol (float, optional): Relative tolerance. Defaults to 1e-02.
        atol (float, optional): Absolute tolerance. Defaults to 1e-04.
    """
    training_mode = False
    if isinstance(framework_model, torch.nn.Module):
        if framework_model.training:
            training_mode = True
            framework_model.eval()
    pytorch_res = framework_model(*inputs)
    if training_mode:
        framework_model.train()

    if isinstance(pytorch_res, (tuple, list)):
        pytorch_res = pytorch_res[0]
    elif isinstance(pytorch_res, OrderedDict):
        pytorch_res = tuple(pytorch_res.values())[0]

    if isinstance(pytorch_res, tf.Tensor):
        pytorch_res = pytorch_res.numpy()
        pytorch_res = torch.from_numpy(pytorch_res)

    assert compare_tensor_to_golden("tvm", golden=pytorch_res, calculated=pybuda_model_results.outputs[0].value(), rtol=rtol, atol=atol)

    logger.debug("Tensors match on output of framework and pybuda")
