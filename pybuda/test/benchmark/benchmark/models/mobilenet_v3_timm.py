# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch
import timm

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["small", "large"])
def mobilenet_v3_timm(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1" 

    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
    os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

    # Set model parameters based on chosen task and model configuration
    model_name = ""
    if config == "small":
        model_name = "hf_hub:timm/mobilenetv3_small_100.lamb_in1k"
        img_res = 224
    elif config == "large":
        model_name = "hf_hub:timm/mobilenetv3_large_100.ra_in1k"
        img_res = 224
    else:
        raise RuntimeError("Unknown config")

    # Load model
    model = timm.create_model(model_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_mobilenet_v3_timm_{config}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
