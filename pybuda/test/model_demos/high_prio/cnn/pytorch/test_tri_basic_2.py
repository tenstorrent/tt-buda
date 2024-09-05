# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import torch

from types import SimpleNamespace

import cv2
import os
import sys
sys.path = list(set(sys.path + ["third_party/confidential_customer_models/internal/tri_basic_2/"]))

from scripts.semseg import resnet34_semseg

from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind


def test_tri_basic_2_sematic_segmentation_pytorch(test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False

    compiler_cfg.balancer_op_override("add_114", "t_stream_shape", (1, 1))  # TM error
    compiler_cfg.balancer_op_override("add_142", "t_stream_shape", (1, 1))  # TM error
    compiler_cfg.balancer_op_override("add_171", "t_stream_shape", (1, 1))  # TM error

    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_op_override(
            "add_156", "t_stream_shape", (1, 1)
        )  # TM error
        compiler_cfg.balancer_op_override(
            "add_185", "t_stream_shape", (1, 1)
        )  # TM error
        compiler_cfg.balancer_op_override(
            "add_200", "t_stream_shape", (1, 1)
        )  # TM error
        compiler_cfg.balancer_op_override(
            "add_214", "t_stream_shape", (1, 1)
        )  # TM error

    elif test_device.arch == pybuda.BackendDevice.Grayskull:
        compiler_cfg.balancer_op_override(
            "add_200", "t_stream_shape", (1, 1)
        )  # TM error
        compiler_cfg.balancer_op_override(
            "add_229", "t_stream_shape", (1, 1)
        )  # TM error
        compiler_cfg.balancer_op_override(
            "conv2d_15.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2",
            "t_stream_shape",
            (10, 1),
        )

    os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"

    # Sample Input
    image_w = 800
    image_h = 800
    image = cv2.imread(
        "third_party/confidential_customer_models/internal/tri_basic_2/files/samples/left.png"
    )
    image = cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    image_tensor = (
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    ).contiguous()

    # Load the model and weights
    hparams = SimpleNamespace(num_classes=24)
    model = resnet34_semseg(hparams)
    state_dict = torch.load(
        "third_party/confidential_customer_models/internal/tri_basic_2/files/weights/basic_semseg.ckpt",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pt_tri_basic_2_semseg", model)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(image_tensor.shape,)],
        inputs=[(image_tensor,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
