# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda

from pybuda._C.backend_api import BackendDevice
from ..common import benchmark_model, generate_test_device
from test.model_demos.models.t5 import generate_t5_past_cache_enc_dec
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["base", "large"])
def t5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):

    compiler_cfg = _get_global_compiler_config()
    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"

    if data_type == "Bfp8_b":
        pybuda.config.configure_mixed_precision(op_type="add", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="subtract", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="reciprocal", output_df=pybuda.DataFormat.Float16_b)

    available_devices = pybuda.detect_available_devices()
    # Determine model variant
    if config == "base":
        variant = "t5-base"
        if available_devices and available_devices[0] == BackendDevice.Grayskull:
            # Hang, this piece of code should be removed when we consume the budabackend fix
            # https://yyz-gitlab.local.tenstorrent.com/tenstorrent/budabackend/-/merge_requests/1684
            return
    elif config == "large":
        variant = "t5-large"
    else:
        raise RuntimeError("Unknown config")

    # Load model
    modules, inputs, other = generate_t5_past_cache_enc_dec(
        generate_test_device(devtype, arch),
        variant,
    )

    targets = tuple()

    return modules, inputs, targets, other


@benchmark_model(configs=["base", "large"])
def flan_t5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):

    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"

    # Determine model variant
    if config == "base":
        variant = "google/flan-t5-base"
    elif config == "large":
        variant = "google/flan-t5-large"
    else:
        raise RuntimeError("Unknown config")
    
    if data_type == "Bfp8_b":
        pybuda.config.configure_mixed_precision(op_type="add", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="subtract", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="reciprocal", output_df=pybuda.DataFormat.Float16_b)

    # Load model
    modules, inputs, other = generate_t5_past_cache_enc_dec(
        generate_test_device(devtype, arch),
        variant,
    )
 
    targets = tuple()

    return modules, inputs, targets, other
