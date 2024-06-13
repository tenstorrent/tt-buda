# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda

from pybuda._C.backend_api import BackendDevice
from ..common import benchmark_model, generate_test_device
from test.model_demos.models.t5 import generate_t5_past_cache_enc_dec
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["base", "large"])
def t5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    from pybuda._C.backend_api import BackendDevice

    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    if data_type == "Fp16_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
        os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"
        # Disable DRAM BW estimates.
        os.environ["PYBUDA_BALANCER_USE_DRAM_BW_ESTIMATES"] = "0"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_EXP_APPROX"] = "1"

    if data_type == "Bfp8_b":
        pybuda.config.configure_mixed_precision(op_type="add", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="subtract", output_df=pybuda.DataFormat.Float16_b)
        pybuda.config.configure_mixed_precision(op_type="reciprocal", output_df=pybuda.DataFormat.Float16_b)

    available_devices = pybuda.detect_available_devices()
    # Determine model variant
    if config == "base":
        variant = "t5-base"

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
def flan_t5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    from pybuda._C.backend_api import BackendDevice

    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    if data_type == "Fp16_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
        os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_EXP_APPROX"] = "1"

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
