# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch
import torch.multiprocessing

from pytorchcv.model_provider import get_model as ptcv_get_model
from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config
from pybuda._C.backend_api import BackendDevice


torch.multiprocessing.set_sharing_strategy("file_system")


@benchmark_model(
    configs=[
        "w18",
        "v2_w18",
        "v2_w30",
        "v2_w32",
        "v2_w40",
        "v2_w44",
        "v2_w48",
        "v2_w64",
    ]
)
def hrnet(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "46" # removing causes hang #2139
    os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"
    if data_type == "Fp16_b":
        os.environ["PYBUDA_RIBBON2_OPTIMIZATION_ITERATIONS"] = "10"

    # Manually enable amp light for Ribbon
    if compiler_cfg.balancer_policy == "Ribbon":
        compiler_cfg.enable_amp_light()

    # Set model parameters based on chosen task and model configuration
    img_res = 224

    if config == "w18":
        model_name = "hrnet_w18_small_v2"
    elif config == "v2_w18":
        model_name = "hrnetv2_w18"
    elif config == "v2_w30":
        model_name = "hrnetv2_w30"
    elif config == "v2_w32":
        model_name = "hrnetv2_w32"
    elif config == "v2_w40":
        model_name = "hrnetv2_w40"
    elif config == "v2_w44":
        model_name = "hrnetv2_w44"
    elif config == "v2_w48":
        model_name = "hrnetv2_w48"
    elif config == "v2_w64":
        model_name = "hrnetv2_w64"
        if data_type == "Bfp8_b":
            if "TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE" not in os.environ:
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{10*1024}"
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                pybuda.config._internal_insert_fj_buffering_nop('add_312', ['add_341'], nop_count=2)
    else:
        raise RuntimeError("Unknown config")

    # Load model
    model = ptcv_get_model(model_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"pt_hrnet_{config}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
