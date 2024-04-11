# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import timm
import torch

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["19", "39", "99"])
def vovnet_v2(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1" 

    os.environ["PYBUDA_DISABLE_EXPLICIT_DRAM_IO"] = "1"
    os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"
    os.environ["PYBUDA_RIBBON2_CONSERVATIVE_OPTIMIZATION_ITERATIONS"] = "10"

    if config == "39" and data_type != "Bfp8_b":
        compiler_cfg.enable_amp_light()

    # Set model parameters based on chosen task and model configuration
    img_res = 224

    model_name = ""
    if config == "19":
        model_name = "ese_vovnet19b_dw"
    elif config == "39":
        model_name = "ese_vovnet39b"
    elif config == "99":
        model_name = "ese_vovnet99b"
    else:
        raise RuntimeError("Unknown config")
    
    # Load model
    model = timm.create_model(model_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_vovnet_v2_{config}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
