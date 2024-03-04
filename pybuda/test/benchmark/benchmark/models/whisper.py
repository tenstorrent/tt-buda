# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import torch
import os

from ..common import benchmark_model, generate_test_device
from pybuda.config import _get_global_compiler_config

from test.model_demos.models.whisper import generate_model_whisper_decoder_past_cache, generate_model_whisper_enc_dec


@benchmark_model(configs=["tiny","small"])
def whisper_decoder(training: bool, config: str, microbatch: int, devtype: str, arch: str):
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
def whisper(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):

    compiler_cfg = _get_global_compiler_config()
    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    # Determine model variant
    if config == "small":
        variant = "openai/whisper-small"
    else:
        raise RuntimeError("Unknown config")
    
    from pybuda._C.backend_api import BackendDevice
    
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

