# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test for Resnet Bottleneck Block
#

import pybuda
import pybuda.op
from pybuda import PyBudaModule

from pybuda.op.nn import Conv2dModule


class TestBottleneckBlock(PyBudaModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        inbetween_channels,
        stride,
        dilation,
        depthwise,
        bias,
    ):
        super().__init__("ResNet Basic Block")

        assert in_channels == out_channels, "Number of input and output channels must be the same"

        self.conv1 = Conv2dModule(
                "conv2d_1",
                in_channels,
                inbetween_channels,
                1,
                stride=stride,
                padding=(1 // 2),  # only padding we support today
                dilation=dilation,
                groups=in_channels if depthwise else 1,
                bias=bias,
            )
        self.conv2 = Conv2dModule(
                "conv2d_2",
                inbetween_channels,
                inbetween_channels,
                3,
                stride=stride,
                padding=(3 // 2),  # only padding we support today
                dilation=dilation,
                groups=inbetween_channels if depthwise else 1,
                bias=bias,
            )
        self.conv3 = Conv2dModule(
                "conv2d_3",
                inbetween_channels,
                out_channels,
                3,
                stride=stride,
                padding=(3 // 2),  # only padding we support today
                dilation=dilation,
                groups=inbetween_channels if depthwise else 1,
                bias=bias,
            )

    def forward(self, activations):

        # Left side of the block, three convolutional layers with relu
        conv1 = self.conv1(activations)
        relu1 = pybuda.op.Relu("relu1", conv1)
        conv2 = self.conv2(relu1)
        relu2 = pybuda.op.Relu("relu2", conv2)
        conv3 = self.conv3(relu2)

        # Right side of the block, just identity tensor, activations 
        # added to convolution from the left side        
        add1 = pybuda.op.Add("out", conv3, activations)

        # Sum of outputs from the left and right sides with applied relu
        relu3 = pybuda.op.Relu("relu3", add1)

        return relu3