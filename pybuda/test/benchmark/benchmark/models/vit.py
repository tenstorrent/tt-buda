# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config
from transformers import ViTForImageClassification


@benchmark_model(configs=["base", "large"])
def vit(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    from pybuda._C.backend_api import BackendDevice

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    if data_type == "Bfp8_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
        os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"
        # Disable NOC BW estimates.
        os.environ["PYBUDA_BALANCER_USE_NOC_BW_ESTIMATES"] = "0"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

    if data_type == "Fp16_b":
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES_APPLY_FILTERING"] = "1"

    if data_type == "Bfp8_b":
        os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
        pybuda.config.configure_mixed_precision(op_type="reciprocal", output_df=pybuda.DataFormat.Float16_b)
        os.environ["PYBUDA_FUSE_DF_OVERRIDE"] = "0"

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
