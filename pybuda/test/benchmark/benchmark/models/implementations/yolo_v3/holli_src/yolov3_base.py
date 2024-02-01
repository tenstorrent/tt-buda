# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
import importlib

from .yolo_layer import *


class Yolov3Base(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_loss_layers(self):
        return [self.yolo_0, self.yolo_1]

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x):
        shape = x.shape
        assert (
            shape[1] == 3 and shape[2] % 32 == 0 and shape[3] % 32 == 0
        ), f"Tensor shape should be [bs, 3, x*32, y*32], was {shape}"
        xb = self.forward_backbone(x)
        return self.forward_yolo(xb)

    def boxes_from_output(self, outputs, conf_thresh=0.25):
        all_boxes = [[] for j in range(outputs[0].size(0))]
        for i, layer in enumerate(self.get_loss_layers()):
            layer_boxes = layer.get_region_boxes(outputs[i], conf_thresh=conf_thresh)
            for j, layer_box in enumerate(layer_boxes):
                all_boxes[j] += layer_box

        return all_boxes

    def predict_img(self, imgs, conf_thresh=0.25):
        self.eval()
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(-1)

        outputs = self.forward(imgs)
        return self.boxes_from_output(outputs, conf_thresh)

    def freeze_backbone(self, requires_grad=False):
        for _, p in self.backbone.named_parameters():
            p.requires_grad = requires_grad

    def unfreeze(self):
        for _, p in self.named_parameters():
            p.requires_grad = True

    def freeze_info(self, print_all=False):
        d = defaultdict(set)
        print("Layer: param.requires_grad")
        for name, param in self.named_parameters():
            if print_all:
                print(f"{name}: {param.requires_grad}")
            else:
                d[name.split(".")[0]].add(param.requires_grad)
        if not print_all:
            for k, v in d.items():
                print(k, ": ", v)

    def load_backbone(self, h5_path):
        state_old = self.state_dict()
        state_new = torch.load(h5_path)

        skipped_layers = []
        for k in list(state_new.keys()):
            if state_old[k].shape != state_new[k].shape:
                skipped_layers.append(k)
                del state_new[k]

        # for k in list(state_dict.keys()):
        #     if k.startswith(('yolo_0_pre.15', 'yolo_1_pre.20')):
        #         del state_dict[k]

        # Renaming some keys if needed for compatibility
        # state_dict = type(state_dict_org)()
        # for k_old in list(state_dict.keys()):
        #     k_new = k_old.replace('backend', 'backbone')
        #     state_dict[k_new] = state_dict_org[k_old]

        return self.load_state_dict(state_new, strict=False), skipped_layers


###################################################################
## Common helper modules


# from fastai.models.darknet import ConvBN
class ConvBN(nn.Module):
    "convolutional layer then batchnorm"

    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # we should never need to set padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        return nn.Upsample(scale_factor=self.stride, mode="nearest")(x)
