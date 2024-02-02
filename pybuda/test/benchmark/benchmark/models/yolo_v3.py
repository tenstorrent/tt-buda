# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pybuda
import torch

from ..common import benchmark_model
from pybuda.config import _get_global_compiler_config
from .implementations.yolo_v3.holli_src import utils
from .implementations.yolo_v3.holli_src.yolo_layer import *
from .implementations.yolo_v3.holli_src.yolov3_tiny import *
from .implementations.yolo_v3.holli_src.yolov3 import *


@benchmark_model(configs=["default", "tiny"])
def yolo_v3(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1" 

    os.environ["PYBUDA_OVERRIDE_INPUT_QUEUE_ENTRIES"] = "32"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    if data_type != "Bfp8_b":
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"

    if data_type == "Bfp8_b":
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

    # TODO: Due to issue tenstorrent/pybuda#1297 
    pybuda.config.override_op_size("_fused_op_1", (2, 2))

    # Set model parameters based on chosen task and model configuration
    config_name = ""
    if config == "default":
        img_res = 512
        # Load model
        model = Yolov3(num_classes=80)
        model.load_state_dict(torch.load('third_party/confidential_customer_models/model_2/pytorch/yolo_v3/weights/yolov3_coco_01.h5', map_location=torch.device('cpu')))
    elif config == "tiny":
        img_res = 512
        # Load model
        model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
        model.load_state_dict(torch.load('third_party/confidential_customer_models/model_2/pytorch/yolo_v3/weights/yolov3_tiny_coco_01.h5'))
    else:
        raise RuntimeError("Unknown config")

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": pybuda.PyTorchModule(f"yolov3_holli_{config}_{compiler_cfg.balancer_policy}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
