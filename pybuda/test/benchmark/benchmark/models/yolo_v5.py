# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import torch
import os
# import yolov5  # use this instead pybuda/test/tvm/cnn/pytorch/tests_C/test_yolov5.py

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config


@benchmark_model(configs=["s", "m"])
def yolo_v5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    compiler_cfg = _get_global_compiler_config()


    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    from pybuda._C.backend_api import BackendDevice
    available_devices = pybuda.detect_available_devices()
    if available_devices[0] == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "49"

    # Temp perf workaround for tenstorrent/bbe#2595
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"

    if data_type == "Fp16_b":
        os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"

    if data_type == "Bfp8_b":
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"
        # Temp workaround for tenstorrent/bbe#2595, output BW is unpredictable.
        os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"

    if available_devices[0] == BackendDevice.Grayskull:
        compiler_cfg.enable_tm_cpu_fallback = True


    # Set model parameters based on chosen task and model configuration
    config_name = ""
    if config == "s":
        config_name = "yolov5s"
        img_res = 320
    elif config == "m":
        config_name = "yolov5m",
        img_res = 640
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = 32  # default

    # Load model
    model = torch.hub.load("ultralytics/yolov5", config_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # The model is implemented with dynamic shapes as it supports various input sizes... Needs to be run with proper
    # input shape on CPU so that the dynamic shapes get resolved properly, before running thru pybuda
    model(inputs[0])

    modules = {"tt": pybuda.PyTorchModule(f"yolov5_{config}_{compiler_cfg.balancer_policy}", model)}

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
