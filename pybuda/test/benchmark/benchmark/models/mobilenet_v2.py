# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config
from transformers import AutoModelForImageClassification


@benchmark_model(configs=["224", "160", "96"])
def mobilenet_v2(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    if data_type != "Bfp8_b":
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"
        os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1" 
        os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"

    if arch == "grayskull":
        os.environ["PYBUDA_MAXIMIZE_SPARSE_UBLOCK"] = "1"
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1" 
        os.environ["PYBUDA_RIBBON2_OPTIMIZATION_ITERATIONS"] = "10" 
        os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"

    if data_type == "Bfp8_b":
        pybuda.config.configure_mixed_precision(name_regex="input.*add.*", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="add", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="depthwise", input_df={1: (pybuda.DataFormat.Float16_b, False),}, output_df=pybuda.DataFormat.Float16_b)

    # Set model parameters based on chosen task and model configuration
    if config == "224":
        model_name = "google/mobilenet_v2_1.0_224"
        img_res = 224
    elif config == "160":
        model_name = "google/mobilenet_v2_0.75_160"
        img_res = 160
    elif config == "96":
        model_name = "google/mobilenet_v2_0.35_96"
        img_res = 96
    else:
        raise RuntimeError("Unknown config")

    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_mobilenet_v2_{config}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
