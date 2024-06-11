# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import torch
import os

from ..common import benchmark_model, generate_test_device
from pybuda.config import _get_global_compiler_config

from test.model_demos.models.whisper import generate_model_whisper_decoder_past_cache, generate_model_whisper_enc_dec


@benchmark_model(configs=["tiny","small"])
def whisper_decoder(training: bool, config: str, microbatch: int, devtype: str, arch: str, math_fidelity: str):
    # Determine model variant
    if config == "tiny":
        variant = "openai/whisper-tiny"
    elif config == "small":
        variant = "openai/whisper-small"
    else:
        raise RuntimeError("Unknown config")
    
    # Load model
    model, inputs, other = generate_model_whisper_decoder_past_cache(
        generate_test_device(devtype, arch),
        variant,
    )
    
    modules = {"tt": model}
    targets = tuple()

    return modules, inputs, targets, other


@benchmark_model(configs=["small"])
def whisper(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    # Determine model variant
    if config == "small":
        variant = "openai/whisper-small"
    else:
        raise RuntimeError("Unknown config")
    
    from pybuda._C.backend_api import BackendDevice

    if data_type == "Fp16_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
        os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"
    
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            pybuda.config.set_epoch_break("conv2d_9.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")
            pybuda.config.override_op_size("conv2d_9.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 12))
    
    # Load model
    model, inputs, other = generate_model_whisper_enc_dec(
        generate_test_device(devtype, arch),
        variant,
    )
    
    modules = {"tt": model}
    targets = tuple()

    return modules, inputs, targets, other

