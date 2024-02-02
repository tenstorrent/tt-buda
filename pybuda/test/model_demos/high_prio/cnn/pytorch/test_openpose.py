# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from collections import OrderedDict

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

import pybuda
from pybuda import VerifyConfig, CompileDepth
from pybuda.verify.config import TestKind, BackendDevice, DataFormat, NebulaGalaxy
from pybuda._C.backend_api import BackendType
from pybuda.verify.backend import verify_module
from test.utils import download_model


def get_image_tensor(sample_path):
    input_image = Image.open(sample_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights[
            ".".join(weights_name.split(".")[1:])
        ]
    return transfered_model_weights


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4],
            )
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class OpenPoseBodyModel(nn.Module):
    def __init__(self):
        super(OpenPoseBodyModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = [
            "conv5_5_CPM_L1",
            "conv5_5_CPM_L2",
            "Mconv7_stage2_L1",
            "Mconv7_stage2_L2",
            "Mconv7_stage3_L1",
            "Mconv7_stage3_L2",
            "Mconv7_stage4_L1",
            "Mconv7_stage4_L2",
            "Mconv7_stage5_L1",
            "Mconv7_stage5_L2",
            "Mconv7_stage6_L1",
            "Mconv7_stage6_L1",
        ]
        blocks = {}
        block0 = OrderedDict(
            [
                ("conv1_1", [3, 64, 3, 1, 1]),
                ("conv1_2", [64, 64, 3, 1, 1]),
                ("pool1_stage1", [2, 2, 0]),
                ("conv2_1", [64, 128, 3, 1, 1]),
                ("conv2_2", [128, 128, 3, 1, 1]),
                ("pool2_stage1", [2, 2, 0]),
                ("conv3_1", [128, 256, 3, 1, 1]),
                ("conv3_2", [256, 256, 3, 1, 1]),
                ("conv3_3", [256, 256, 3, 1, 1]),
                ("conv3_4", [256, 256, 3, 1, 1]),
                ("pool3_stage1", [2, 2, 0]),
                ("conv4_1", [256, 512, 3, 1, 1]),
                ("conv4_2", [512, 512, 3, 1, 1]),
                ("conv4_3_CPM", [512, 256, 3, 1, 1]),
                ("conv4_4_CPM", [256, 128, 3, 1, 1]),
            ]
        )

        # Stage 1
        block1_1 = OrderedDict(
            [
                ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
            ]
        )

        block1_2 = OrderedDict(
            [
                ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
            ]
        )
        blocks["block1_1"] = block1_1
        blocks["block1_2"] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks["block%d_1" % i] = OrderedDict(
                [
                    ("Mconv1_stage%d_L1" % i, [185, 128, 7, 1, 3]),
                    ("Mconv2_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv3_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv4_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv5_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv6_stage%d_L1" % i, [128, 128, 1, 1, 0]),
                    ("Mconv7_stage%d_L1" % i, [128, 38, 1, 1, 0]),
                ]
            )

            blocks["block%d_2" % i] = OrderedDict(
                [
                    ("Mconv1_stage%d_L2" % i, [185, 128, 7, 1, 3]),
                    ("Mconv2_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv3_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv4_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv5_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv6_stage%d_L2" % i, [128, 128, 1, 1, 0]),
                    ("Mconv7_stage%d_L2" % i, [128, 19, 1, 1, 0]),
                ]
            )

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks["block1_1"]
        self.model2_1 = blocks["block2_1"]
        self.model3_1 = blocks["block3_1"]
        self.model4_1 = blocks["block4_1"]
        self.model5_1 = blocks["block5_1"]
        self.model6_1 = blocks["block6_1"]

        self.model1_2 = blocks["block1_2"]
        self.model2_2 = blocks["block2_2"]
        self.model3_2 = blocks["block3_2"]
        self.model4_2 = blocks["block4_2"]
        self.model5_2 = blocks["block5_2"]
        self.model6_2 = blocks["block6_2"]

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class OpenPoseHandModel(nn.Module):
    def __init__(self):
        super(OpenPoseHandModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = [
            "conv6_2_CPM",
            "Mconv7_stage2",
            "Mconv7_stage3",
            "Mconv7_stage4",
            "Mconv7_stage5",
            "Mconv7_stage6",
        ]
        # stage 1
        block1_0 = OrderedDict(
            [
                ("conv1_1", [3, 64, 3, 1, 1]),
                ("conv1_2", [64, 64, 3, 1, 1]),
                ("pool1_stage1", [2, 2, 0]),
                ("conv2_1", [64, 128, 3, 1, 1]),
                ("conv2_2", [128, 128, 3, 1, 1]),
                ("pool2_stage1", [2, 2, 0]),
                ("conv3_1", [128, 256, 3, 1, 1]),
                ("conv3_2", [256, 256, 3, 1, 1]),
                ("conv3_3", [256, 256, 3, 1, 1]),
                ("conv3_4", [256, 256, 3, 1, 1]),
                ("pool3_stage1", [2, 2, 0]),
                ("conv4_1", [256, 512, 3, 1, 1]),
                ("conv4_2", [512, 512, 3, 1, 1]),
                ("conv4_3", [512, 512, 3, 1, 1]),
                ("conv4_4", [512, 512, 3, 1, 1]),
                ("conv5_1", [512, 512, 3, 1, 1]),
                ("conv5_2", [512, 512, 3, 1, 1]),
                ("conv5_3_CPM", [512, 128, 3, 1, 1]),
            ]
        )

        block1_1 = OrderedDict(
            [("conv6_1_CPM", [128, 512, 1, 1, 0]), ("conv6_2_CPM", [512, 22, 1, 1, 0])]
        )

        blocks = {}
        blocks["block1_0"] = block1_0
        blocks["block1_1"] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks["block%d" % i] = OrderedDict(
                [
                    ("Mconv1_stage%d" % i, [150, 128, 7, 1, 3]),
                    ("Mconv2_stage%d" % i, [128, 128, 7, 1, 3]),
                    ("Mconv3_stage%d" % i, [128, 128, 7, 1, 3]),
                    ("Mconv4_stage%d" % i, [128, 128, 7, 1, 3]),
                    ("Mconv5_stage%d" % i, [128, 128, 7, 1, 3]),
                    ("Mconv6_stage%d" % i, [128, 128, 1, 1, 0]),
                    ("Mconv7_stage%d" % i, [128, 22, 1, 1, 0]),
                ]
            )

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks["block1_0"]
        self.model1_1 = blocks["block1_1"]
        self.model2 = blocks["block2"]
        self.model3 = blocks["block3"]
        self.model4 = blocks["block4"]
        self.model5 = blocks["block5"]
        self.model6 = blocks["block6"]

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


variants = [
    "body_basic",
    "hand_basic",
]

def generate_model_openpose_posdet_custom_pytorch(test_device, variant):
    # Init config
    compiler_cfg = pybuda.config._get_global_compiler_config()
    
    if test_device.arch == BackendDevice.Grayskull:
        # Limit to BE Golden verify as hang occures due to the fork-join buffering
        # tenstorrent/pybuda#880
        compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    if variant == "body_basic" and test_device.arch == BackendDevice.Grayskull:
        # Possibilities of finding out better way of handling extra blob gen size
        # tenstorrent/pybuda#881
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{12*1024}"
        
    if variant == "hand_basic" and test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{96*1024}"

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    # Data type errors while using AMP = 1
    # tenstorrent/pybuda#856
    # compiler_cfg.amp_level = 2

    # No valid grids found for many conv ops
    # tenstorrent/pybuda#872
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Load model
    if variant == "body_basic":
        model_path = "third_party/confidential_customer_models/model_2/pytorch/openpose/weights/body_pose_model.pth"
        framework_model = OpenPoseBodyModel()
        sample_path = (
            "third_party/confidential_customer_models/model_2/pytorch/openpose/samples/body.jpeg"
        )

    elif variant == "hand_basic":
        model_path = "third_party/confidential_customer_models/model_2/pytorch/openpose/weights/hand_pose_model.pth"
        framework_model = OpenPoseHandModel()
        sample_path = (
            "third_party/confidential_customer_models/model_2/pytorch/openpose/samples/hand.jpeg"
        )
    framework_model_dict = transfer(framework_model, torch.load(model_path))
    framework_model.load_state_dict(framework_model_dict)
    pybuda_model = pybuda.PyTorchModule("open_pose_" + variant + "_pt", framework_model)

    # Load & pre-process image
    img_tensor = get_image_tensor(sample_path)

    # Sanity run
    cpu_out = framework_model(img_tensor)
    
    return pybuda_model, [img_tensor], {}


@pytest.mark.parametrize("variant", variants)
def test_openpose_basic(variant, test_device):
    model, inputs, _ = generate_model_openpose_posdet_custom_pytorch(
        test_device, variant,
    )

    # Verify
    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            pcc=0.9,
        ),
    )


def generate_model_openpose_posdet_osmr_pytorch(test_device, variant):
    if (
        test_device.arch == BackendDevice.Grayskull
        and variant == "lwopenpose2d_mobilenet_cmupan_coco"
    ):
        pytest.skip("Grayskull failing with data mismatch PCC = 0.5567322296627039")

    if (
        test_device.arch == BackendDevice.Grayskull
        and variant == "lwopenpose3d_mobilenet_cmupan_coco"
    ):
        pytest.skip("Grayskull failing with data mismatch PCC = 0.7433900704259362")

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16

    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()
    pybuda_model = pybuda.PyTorchModule("openpose_" + variant + "_pt", framework_model)

    # Load & pre-process image
    sample_path = "third_party/confidential_customer_models/model_2/pytorch/openpose/samples/body.jpeg"
    img_tensor = get_image_tensor(sample_path)

    # Sanity run
    cpu_out = framework_model(img_tensor)
    
    return pybuda_model, [img_tensor], {}


variants = [
    "lwopenpose2d_mobilenet_cmupan_coco",
    "lwopenpose3d_mobilenet_cmupan_coco",
]


@pytest.mark.parametrize("variant", variants)
def test_openpose_osmr(variant, test_device):
    model, inputs, _ = generate_model_openpose_posdet_osmr_pytorch(
        test_device, variant,
    )

    # Verify
    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.85,
        ),
    )
