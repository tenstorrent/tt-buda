# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import torch

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config
from pytorchcv.model_provider import get_model as ptcv_get_model


@benchmark_model(configs=["27s", "39", "57"])
def vovnet_v1(training: bool, config: str, microbatch: int, devtype: str, arch: str):
    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    # Set model parameters based on chosen task and model configuration
    img_res = 224

    model_name = ""
    if config == "27s":
        model_name = "vovnet27s"
    elif config == "39":
        model_name = "vovnet39"
    elif config == "57":
        model_name = "vovnet57"
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = 32  # default

    # Load model
    model = ptcv_get_model(model_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_vovnet_v1_{config}_{compiler_cfg.balancer_policy}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
