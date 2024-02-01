# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from .yolo_layer import *
from .yolov3_base import *


class Yolov3(Yolov3Base):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = Darknet([1, 2, 8, 8, 4])

        anchors_per_region = 3
        self.yolo_0_pre = Yolov3UpsamplePrep(
            [512, 1024], 1024, anchors_per_region * (5 + num_classes)
        )
        self.yolo_0 = YoloLayer(
            anchors=[(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)],
            stride=32,
            num_classes=num_classes,
        )

        self.yolo_1_c = ConvBN(512, 256, 1)
        self.yolo_1_prep = Yolov3UpsamplePrep(
            [256, 512], 512 + 256, anchors_per_region * (5 + num_classes)
        )
        self.yolo_1 = YoloLayer(
            anchors=[(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)],
            stride=16,
            num_classes=num_classes,
        )

        self.yolo_2_c = ConvBN(256, 128, 1)
        self.yolo_2_prep = Yolov3UpsamplePrep(
            [128, 256], 256 + 128, anchors_per_region * (5 + num_classes)
        )
        self.yolo_2 = YoloLayer(
            anchors=[(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)],
            stride=8,
            num_classes=num_classes,
        )

    def get_loss_layers(self):
        return [self.yolo_0, self.yolo_1, self.yolo_2]

    def forward_yolo(self, xb):
        x, y0 = self.yolo_0_pre(xb[-1])

        x = self.yolo_1_c(x)
        x = nn.Upsample(scale_factor=2, mode="nearest")(x)
        x = torch.cat([x, xb[-2]], 1)
        x, y1 = self.yolo_1_prep(x)

        x = self.yolo_2_c(x)
        x = nn.Upsample(scale_factor=2, mode="nearest")(x)
        x = torch.cat([x, xb[-3]], 1)
        x, y2 = self.yolo_2_prep(x)

        return [y0, y1, y2]


###################################################################
## Backbone and helper modules


class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in // 2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class Darknet(nn.Module):
    def __init__(self, num_blocks, start_nf=32):
        super().__init__()
        nf = start_nf
        self.base = ConvBN(3, nf, kernel_size=3, stride=1)  # , padding=1)
        self.layers = []
        for i, nb in enumerate(num_blocks):
            # dn_layer = make_group_layer(nf, nb, stride=(1 if i==-1 else 2))
            dn_layer = self.make_group_layer(nf, nb, stride=2)
            self.add_module(f"darknet_{i}", dn_layer)
            self.layers.append(dn_layer)
            nf *= 2

    def make_group_layer(self, ch_in, num_blocks, stride=2):
        layers = [ConvBN(ch_in, ch_in * 2, stride=stride)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = [self.base(x)]
        for l in self.layers:
            y.append(l(y[-1]))
        return y


class Yolov3UpsamplePrep(nn.Module):
    def __init__(self, filters_list, in_filters, out_filters):
        super().__init__()
        self.branch = nn.ModuleList(
            [
                ConvBN(in_filters, filters_list[0], 1),
                ConvBN(filters_list[0], filters_list[1], kernel_size=3),
                ConvBN(filters_list[1], filters_list[0], kernel_size=1),
                ConvBN(filters_list[0], filters_list[1], kernel_size=3),
                ConvBN(filters_list[1], filters_list[0], kernel_size=1),
            ]
        )
        self.for_yolo = nn.ModuleList(
            [
                ConvBN(filters_list[0], filters_list[1], kernel_size=3),
                nn.Conv2d(
                    filters_list[1],
                    out_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            ]
        )

    def forward(self, x):
        for m in self.branch:
            x = m(x)
        branch_out = x
        for m in self.for_yolo:
            x = m(x)
        return branch_out, x
