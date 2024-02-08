# Copyright 2023 Toyota Research Institute.  All rights reserved.

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels, stride=1, dilation_rate=1):
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation_rate,
                               dilation=dilation_rate, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate,
                               bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)

        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        outputs = outputs + shortcut
        outputs = self.activation(outputs)
        return outputs


def resnet_group(*, block_func, in_channels, out_channels, stride, num_blocks, dilation_rates=[1]):
    assert len(dilation_rates) > 0

    residual_blocks = [
        block_func(in_channels=in_channels, out_channels=out_channels, stride=stride, dilation_rate=dilation_rates[0])
    ]
    for idx in range(1, num_blocks):
        residual_blocks.append(block_func(in_channels=out_channels, out_channels=out_channels, stride=1,
                                          dilation_rate=dilation_rates[idx % len(dilation_rates)]))
    return nn.Sequential(*residual_blocks)


class Fpn(nn.Module):

    def __init__(self, *, in_channels, out_channels):
        super().__init__()

        idxs = []
        convs = []
        for idx, channels in enumerate(in_channels):
            idxs.append(idx)
            convs.append(nn.Conv2d(channels, out_channels, kernel_size=1, bias=True))
        self.idxs = idxs[::-1]
        self.convs = nn.ModuleList(convs[::-1])

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, group_outputs: List[torch.Tensor]):
        outputs = None
        for idx, module in enumerate(self.convs):
            current = module(group_outputs[self.idxs[idx]])
            if outputs is None:
                outputs = current
            else:
                outputs = self.upsample2(outputs) + current

        return outputs


class BasicResNet(nn.Module):

    def __init__(self, hparams, *, num_blocks, num_channels, dilation_rates):
        super().__init__()
        assert len(num_blocks) == 4
        assert len(num_channels) == len(num_blocks)
        assert len(dilation_rates) == len(num_blocks)

        self.num_channels = num_channels

        self.conv_in = nn.Conv2d(3, num_channels[0], kernel_size=7, padding=3, stride=2, bias=False)
        self.norm_in = nn.BatchNorm2d(num_channels[0])
        self.activation_in = nn.ReLU(inplace=True)
        self.pool_in = nn.MaxPool2d(kernel_size=2)

        self.group1 = resnet_group(block_func=BasicResidualBlock, in_channels=num_channels[0],
                                   out_channels=num_channels[0], stride=1, num_blocks=num_blocks[0],
                                   dilation_rates=dilation_rates[0])
        self.group2 = resnet_group(block_func=BasicResidualBlock, in_channels=num_channels[0],
                                   out_channels=num_channels[1], stride=2, num_blocks=num_blocks[1],
                                   dilation_rates=dilation_rates[1])
        self.group3 = resnet_group(block_func=BasicResidualBlock, in_channels=num_channels[1],
                                   out_channels=num_channels[2], stride=2, num_blocks=num_blocks[2],
                                   dilation_rates=dilation_rates[2])
        self.group4 = resnet_group(block_func=BasicResidualBlock, in_channels=num_channels[2],
                                   out_channels=num_channels[3], stride=2, num_blocks=num_blocks[3],
                                   dilation_rates=dilation_rates[3])

        self.head = Fpn(in_channels=num_channels, out_channels=hparams.num_classes)

        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def get_output_channels(self):
        return self.num_channels

    def forward(self, inputs):
        _, _, h, w = inputs.shape

        vpad = math.ceil(h / 32) * 32 - h
        top_pad = vpad // 2
        bottom_pad = vpad - top_pad
        hpad = math.ceil(w / 32) * 32 - w
        left_pad = hpad // 2
        right_pad = hpad - left_pad

        inputs = F.pad(inputs, (left_pad, right_pad, top_pad, bottom_pad))

        outputs = self.pool_in(self.activation_in(self.norm_in(self.conv_in(inputs))))
        group1_outputs = self.group1(outputs)
        group2_outputs = self.group2(group1_outputs)
        group3_outputs = self.group3(group2_outputs)
        group4_outputs = self.group4(group3_outputs)

        outputs = [group1_outputs, group2_outputs, group3_outputs, group4_outputs]
        logits = self.upsample(self.head(outputs))

        logits = logits[:, :, top_pad:top_pad + h, left_pad:left_pad + w]

        return logits


def resnet34_semseg(hparams):
    return BasicResNet(hparams,
                       num_blocks=[3, 4, 6, 3],
                       num_channels=[64, 128, 256, 512],
                       dilation_rates=[[1], [1], [1, 1, 2, 5, 9, 17], [1]])
