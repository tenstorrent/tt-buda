# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config
from transformers import ViTForImageClassification


@benchmark_model(configs=["base", "large"])
def vit(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"
        os.environ["PYBUDA_RIBBON2_OPTIMIZATION_ITERATIONS"] = "10"

        # These are about to be enabled by default.
        #
        if data_type != "Bfp8_b":
            os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"
            os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
            os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
            os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES_APPLY_FILTERING"] = "1"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"

    if data_type == "Bfp8_b":
        pybuda.config.configure_mixed_precision(op_type="reciprocal", output_df=pybuda.DataFormat.Float16_b)
        os.environ["PYBUDA_TEMP_BALANCER_MODEL_PCIE_BW"] = "0"
        os.environ["PYBUDA_TEMP_DISABLE_FJ_NOP_SCHEDULE_FIX"] = "1"

    # Set model parameters based on chosen task and model configuration
    img_res = 224

    if config == "base":
        model_name = "google/vit-base-patch16-224"
    elif config == "large":
        model_name = "google/vit-large-patch16-224"
    else:
        raise RuntimeError("Unknown config")

    # Load model
    model = ViTForImageClassification.from_pretrained(model_name)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_vit_{config}_{compiler_cfg.balancer_policy}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
