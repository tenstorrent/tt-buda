# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch
import timm

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["224"])
def inception_v4(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True


    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"

    if data_type == "Bfp8_b":
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"

    # Set model parameters based on chosen task and model configuration
    if config == "224":
        model_name = "inception_v4"
        img_res = 224  # https://github.com/huggingface/pytorch-image-models/blob/main/train.py#L122
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = 32  # default

    # Load model
    model = timm.create_model(model_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_inception_v4_{config}_{compiler_cfg.balancer_policy}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
