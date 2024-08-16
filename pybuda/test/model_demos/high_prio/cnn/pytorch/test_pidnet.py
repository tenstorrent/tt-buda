import pybuda
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pytest
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendDevice

from pybuda import VerifyConfig
import sys
sys.path.append("third_party/confidential_customer_models/internal/pidnet/scripts")
from model_pidnet import update_model_config, get_seg_model


variants = ["pidnet_s", "pidnet_m", "pidnet_l"]


@pytest.mark.parametrize("variant", variants)
def test_pidnet_pytorch(variant, test_device):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    os.environ["PYBUDA_RIBBON2"] = "1"

    # Load and pre-process image
    image_path = "third_party/confidential_customer_models/internal/pidnet/files/samples/road_scenes.png"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    image = image.transpose((2, 0, 1))
    input_image = torch.unsqueeze(torch.tensor(image), 0)

    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        if variant == "pidnet_s":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "217088"
            compiler_cfg.balancer_op_override(
                "conv2d_214.dc.reshape.12.dc.sparse_matmul.1.lc2", "t_stream_shape", (1, 4)
            )
            compiler_cfg.amp_level = 1
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone132.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone107_operand_commute_clone134.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone513_operand_commute_clone607.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.place_on_new_epoch("conv2d_960.dc.reshape.12.dc.sparse_matmul.10.lc2")
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone605.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.place_on_new_epoch("resize2d_353.dc.reshape.5.dc.sparse_matmul.10.lc2")

        elif variant == "pidnet_m":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "335872"
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone132.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone107_operand_commute_clone134.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 32),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone513_operand_commute_clone607.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone605.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone29112_operand_commute_clone29139.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone29137.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone29518_operand_commute_clone29612.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone29610.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )

        elif variant == "pidnet_l":
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.place_on_new_epoch("avg_pool2d_1116.dc.conv2d.1.dc.depthwise.10")
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "229376"
            compiler_cfg.balancer_op_override(
                "conv2d_493.dc.conv2d.5.dc.reshape.0_operand_commute_clone129.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_493.dc.conv2d.5.dc.reshape.0_operand_commute_clone107_operand_commute_clone131.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_917.dc.reshape.0_operand_commute_clone581_operand_commute_clone668.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_917.dc.reshape.0_operand_commute_clone666.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 16),
            )

    if test_device.arch == pybuda.BackendDevice.Blackhole:
        if variant == "pidnet_s":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "217088"
            compiler_cfg.balancer_op_override(
                "conv2d_214.dc.reshape.12.dc.sparse_matmul.1.lc2", "t_stream_shape", (1, 4)
            )
            compiler_cfg.amp_level = 1
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone132.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_377.dc.conv2d.5.dc.reshape.0_operand_commute_clone107_operand_commute_clone134.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone513_operand_commute_clone607.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.place_on_new_epoch("conv2d_960.dc.reshape.12.dc.sparse_matmul.10.lc2")
            compiler_cfg.balancer_op_override(
                "conv2d_1010.dc.reshape.0_operand_commute_clone605.dc.sparse_matmul.4.lc2",
                "t_stream_shape",
                (1, 8),
            )
            compiler_cfg.place_on_new_epoch("resize2d_353.dc.reshape.5.dc.sparse_matmul.10.lc2")

    # Load model
    cfg_model_pretrained, cfg_model_state_file = update_model_config(variant)
    model = get_seg_model(variant, cfg_model_pretrained, imgnet_pretrained=True)
    pretrained_dict = torch.load(cfg_model_state_file, map_location=torch.device("cpu"))

    if "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    pcc = 0.98 if variant == "pidnet_s" and test_device.arch == BackendDevice.Wormhole_B0 else 0.99

    # Verify
    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)
    verify_module(
        tt_model,
        input_shapes=(input_image.shape),
        inputs=[(input_image)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )
