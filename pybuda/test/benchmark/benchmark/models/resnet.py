# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import onnx
import torch
import torch.nn as nn
from typing import Callable, Optional, Type

import pybuda
from pybuda import PyTorchModule, OnnxModule
from pybuda.config import _get_global_compiler_config
from ..common import benchmark_model
from transformers import ResNetForImageClassification

@benchmark_model(configs=["resnet18", "resnet50"])
def resnet(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"
    os.environ["PYBUDA_ALLOW_MULTICOLUMN_SPARSE_MATMUL"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

    if data_type == "Fp16_b":
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES_APPLY_FILTERING"] = "1"

    if data_type == "Bfp8_b":
        pybuda.config.configure_mixed_precision(name_regex="input.*add.*", output_df=pybuda.DataFormat.Float16_b)

    # Set model parameters based on chosen task and model configuration
    if config == "resnet18":
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
        module = PyTorchModule("resnet50", model)
    elif config == "resnet50":
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        module = PyTorchModule("resnet50", model)
    else:
        raise RuntimeError("Unknown config")

    input_shape = (microbatch, 3, 224, 224)
    inputs = [torch.rand(*input_shape)]

    models = { "tt": module }
    if training:
        models["cpu-loss"] = PyTorchModule("l1loss", torch.nn.L1Loss())

    targets = tuple()
    if training:
        targets = [torch.rand(1, 100)]

    return models, inputs, targets, {}

@benchmark_model(configs=["resnet50"])
def resnet_quant(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"

    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_DISABLE_FUSE_OPS"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Set model parameters based on chosen task and model configuration
    if config == "resnet50":
        # Download ONNX model
        save_path = "third_party/confidential_customer_models/quantized/ResNet50-v1.5-Int8.onnx"
        if not os.path.exists(save_path):
            raise RuntimeError("Model not found")

        # LOAD ONNX model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        module = OnnxModule(
            "onnx_quantized_ResNet50",
            onnx_model,
            save_path,
        )
    else:
        raise RuntimeError("Unknown config")

    input_shape = (microbatch, 3, 224, 224)
    inputs = [torch.rand(*input_shape)]

    models = { "tt": module }
    if training:
        raise RuntimeError("Quantized models do not support training.")

    targets = tuple()
    return models, inputs, targets, {}

# SPDX-FileCopyrightText: Copyright (c) 2016 Soumith Chintala
#
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert norm_layer is not None
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResnetHead(nn.Module):
    def __init__(self, inplanes: int, norm_layer: Type[nn.Module] = nn.BatchNorm2d, maxpool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.has_maxpool = maxpool
        if self.has_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.has_maxpool:
            x = self.maxpool(x)
        return x


class Unconv(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-3], x.shape[-1] * x.shape[-2])
        x = x.transpose(-2, -1)
        x = x.sum(-2, keepdim=True)
        return x


class Toconv(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        x = x.transpose(2, 3)
        dim = int(math.sqrt(x.shape[-1]))
        x = x.reshape(-1, x.shape[-2], dim, dim)
        return x


class ToconvBroadcast(nn.Module):
    def __init__(self, repeat_factor, narrow_r):
        super().__init__()
        self.repeat_factor = repeat_factor
        self.narrow_r = narrow_r
        self.to_conv = Toconv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(self.repeat_factor)
        x = x.narrow(2, 0, self.narrow_r)
        return self.to_conv(x)


def _make_resnet_layer(
    block: Bottleneck,
    inplanes: int,
    planes: int,
    blocks: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    previous_dilation: int = 1,
    base_width: int = 64,
    norm_layer = nn.BatchNorm2d,
    to_conv = Toconv,
    un_conv = Unconv,
) -> nn.Sequential:
    downsample = None
    if dilation > 1:
        dilation *= stride
        stride = 1
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer
        )
    )
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return nn.Sequential(to_conv(), *layers, un_conv())


@benchmark_model(configs=["head", "layer1", "layer2", "layer3", "layer4"])
def resnet50_layer(training: bool, config: str, microbatch: int, devtype: str, arch: str):
    layer = config

    compiler_cfg = _get_global_compiler_config()
    # verify_cfg.verify_pybuda_codegen_vs_framework = False # hacking 7x7 to 1x1 will cause mismatches

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    # compiler_cfg.scheduler_policy = "LongestPath"


    if layer == "head":
        if arch == "wormhole_b0":
            compiler_cfg.enable_auto_transposing_placement = True

            fracture_factor = 4
            fractured_conv_sparse_mms = [f"conv2d_0.dc.conv2d.3.dc.conv2d.{1 + i * 2}.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2" for i in range(fracture_factor)]
            fractured_conv_dense_mms = [f"conv2d_0.dc.conv2d.3.dc.conv2d.{1 + i * 2}.dc.matmul.11" for i in range(fracture_factor)]

            pybuda.insert_nop(
                "input_1",
                fractured_conv_sparse_mms,
                hoist_tms=True)

            pybuda.config.override_op_size("buffer_0_input_1_conv2d_0.dc.conv2d.3.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (7, 1))
            pybuda.config.override_t_stream_shape("buffer_0_input_1_conv2d_0.dc.conv2d.3.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))

            pybuda.config.override_multi_op_fracture_factor("conv2d_0.dc.conv2d.3", fracture_factor)
            # pybuda.config.override_multi_op_fracture_factor("conv2d_0", 7)
            for sparse_mm in fractured_conv_sparse_mms:
                pybuda.config.override_op_size(sparse_mm, (7, 1))
                pybuda.config.override_t_stream_shape(sparse_mm, (2, 1))
            for dense_mm in fractured_conv_dense_mms:
                pybuda.config.override_op_size(dense_mm, (7, 1))
                pybuda.config.override_t_stream_shape(dense_mm, (2, 1))
                pybuda.config.override_u_kt(dense_mm, 1)

            pybuda.config.set_epoch_break(["_fused_op_0"])
        else:
            pybuda.config.override_op_size("conv2d_0.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 4))
            # pybuda.config.override_fracture_factor("conv2d_0.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", 4)
            pybuda.config.override_op_size("conv2d_0.dc.conv2d.3.dc.matmul.11", (2, 1))
            pybuda.config.override_op_size("max_pool2d_2.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", (2, 2))

    set_env_common(arch)

    if  microbatch == 0:
        microbatch = 32 # default

    input_mode = "none"
    output_mode = "none"
    maxpool = True
    batch_norm = True
    norm_layer = nn.BatchNorm2d if batch_norm else nn.Identity

    input_shape = {
        "head": (microbatch, 3, 224, 224),
        "layer1": (microbatch, 64, 56, 56),
        "layer2": (microbatch, 256, 56, 56),
        "layer3": (microbatch, 512, 28, 28),
        "layer4": (microbatch, 1024, 14, 14),
    }[layer]

    if input_mode == "pre_reshape":
        input_shape = (input_shape[0], 1, input_shape[2] * input_shape[3], input_shape[1])
    elif input_mode == "broadcast":
        repeat_factor = [1, 1, (input_shape[2] * input_shape[3] + 31) // 32, (input_shape[1] + 31) // 32]
        narrow_r = input_shape[2] * input_shape[3]
        input_shape = [input_shape[0], 1, 32, 32]

    to_conv = {
        "none": nn.Identity,
        "pre_reshape": Toconv,
        "broadcast": lambda: ToconvBroadcast(repeat_factor, narrow_r),
    }[input_mode]

    un_conv = {
        "none": nn.Identity,
        "reduce": Unconv,
    }[output_mode]

    blocks = [3, 4, 6, 3]
    make_model = {
        "head": lambda: nn.Sequential(to_conv(), ResnetHead(64, norm_layer=norm_layer, maxpool=maxpool), un_conv()),
        "layer1": lambda: _make_resnet_layer(Bottleneck, 64, 64, blocks[0], to_conv=to_conv, norm_layer=norm_layer),
        "layer2": lambda: _make_resnet_layer(Bottleneck, 256, 128, blocks[1], stride=2, to_conv=to_conv, norm_layer=norm_layer),
        "layer3": lambda: _make_resnet_layer(Bottleneck, 512, 256, blocks[2], stride=2, to_conv=to_conv, norm_layer=norm_layer),
        "layer4": lambda: _make_resnet_layer(Bottleneck, 1024, 512, blocks[3], stride=2, to_conv=to_conv, norm_layer=norm_layer),
    }[layer]

    inputs = [
            torch.rand(*input_shape)
        ]

    model = make_model()
    module = PyTorchModule(f"resnet50_{layer}", model)

    models = { "tt": module }
    if training:
        models["cpu-loss"] = PyTorchModule("l1loss", torch.nn.L1Loss())

    targets = tuple()
    if training:
        targets = [torch.rand(1, 100)]

    return models, inputs, targets, {}
