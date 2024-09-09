# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda, os
import pytest
from torchvision import transforms
import requests
from PIL import Image
import onnx
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendDevice

variants = ["ddrnet23s", "ddrnet23", "ddrnet39"]


@pytest.mark.parametrize("variant", variants)
def test_ddrnet(variant, test_device):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Wormhole_B0:
        # These overrides are planned to be ON by default
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

    if test_device.arch == BackendDevice.Grayskull:
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

    # STEP 2: # Create PyBuda module from onnx weights
    model_name = f"{variant}_onnx"

    load_path = f"third_party/confidential_customer_models/internal/ddrnet/files/onnx/{variant}.onnx"

    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule(model_name, model, load_path)

    # STEP 3: Prepare input
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    img_tensor = input_tensor.unsqueeze(0)

    verify_module(
        tt_model,
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=(0.98 if test_device.arch == BackendDevice.Grayskull and variant != "ddrnet23s" else 0.99),
        ),
    )


variants = ["ddrnet_23_slim_1024", "ddrnet23_cityscapes"]


@pytest.mark.parametrize("variant", variants)
def test_ddrnet_semantic_segmentation_onnx(variant, test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Wormhole_B0:
        if variant == "ddrnet_23_slim_1024":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "36864"
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone931.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone925.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11803.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11809.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11986.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 16))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11980.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11872.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11866.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))

        elif variant == "ddrnet23_cityscapes":
            compiler_cfg.balancer_op_override("conv2d_213.dc.conv2d.5.dc.reshape.0_operand_commute_clone1044.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 16))
            compiler_cfg.balancer_op_override("conv2d_213.dc.conv2d.5.dc.reshape.0_operand_commute_clone1050.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 16))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "153600"
            os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
            os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
            os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

    elif test_device.arch == BackendDevice.Grayskull:
        if variant == "ddrnet_23_slim_1024":
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone931.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 32))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "24576"
            compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone11915.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 32))

    elif test_device.arch == BackendDevice.Blackhole:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{32*1024}"
        compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone931.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))
        compiler_cfg.balancer_op_override("conv2d_197.dc.conv2d.5.dc.reshape.0_operand_commute_clone925.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 8))

    # Load and validate the model
    if variant == "ddrnet_23_slim_1024":
        load_path = f"third_party/confidential_customer_models/customer/model_0/files/cnn/ddrnet/{variant}.onnx"
    else:
        load_path = f"third_party/confidential_customer_models/internal/ddrnet/files/onnx/{variant}.onnx"
    model = onnx.load(load_path)
    onnx.checker.check_model(model)
    model_name = f"onnx_{variant}"
    tt_model = pybuda.OnnxModule(model_name, model, load_path)

    # Prepare input
    image_path = "third_party/confidential_customer_models/internal/ddrnet/files/samples/road_scenes.png"
    input_image = Image.open(image_path)
    if variant == "ddrnet_23_slim_1024":
        input_image = transforms.Resize((1024, 1024))(input_image)
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Inference
    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
