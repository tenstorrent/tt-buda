# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch
import timm

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["224"])
def mobilenet_v2_timm(training: bool, config: str, microbatch: int, devtype: str, arch: str):
    compiler_cfg = _get_global_compiler_config()
    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    # Set model parameters based on chosen task and model configuration
    if config != "224":
        raise RuntimeError("Unknown config")
    model_name = "mobilenetv2_100"
    img_res = 224

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

    modules = {"tt": pybuda.PyTorchModule(f"pt_mobilenet_v2_timm_{config}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
