# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
from pybuda.verify import verify_module, VerifyConfig, TestKind

class ResnetBottleneck(pybuda.PyBudaModule):
    def __init__(
        self,
        name: str,
        ch_lo: int,
        ch_hi: int,
        use_skip: bool,
    ):
        super().__init__(name)

        self.use_skip = use_skip
        self.use_relu = True

        # right branch
        self.conv_r0 = pybuda.op.nn.Conv2dModule(
            name=name + "_r0",
            in_channels=ch_hi,
            out_channels=ch_lo,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=False,
        )
        self.conv_r1 = pybuda.op.nn.Conv2dModule(
            name=name + "_r1",
            in_channels=ch_lo,
            out_channels=ch_lo,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=False,
        )
        self.conv_r2 = pybuda.op.nn.Conv2dModule(
            name=name + "_r2",
            in_channels=ch_lo,
            out_channels=ch_hi,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=False,
        )

    def set_relu(self, use_relu: bool):
        self.use_relu = use_relu

    def forward(self, x):
        r = pybuda.op.Relu(f"", self.conv_r0(x))
        r = pybuda.op.Relu(f"", self.conv_r1(r))
        r = self.conv_r2(r)

        if self.use_skip:
            r = pybuda.op.Add(f"", x, r)

        if self.use_relu:
            r = pybuda.op.Relu(f"", r)

        return r


def test_bottleneck(test_index, random_seeds, test_device):
    ch_lo = 128
    ch_hi = 512
    use_skip = True
    verify_module(ResnetBottleneck(name=f"resnet_bottleneck_{test_index}", ch_lo=ch_lo, ch_hi=ch_hi, use_skip=use_skip), [(4, 512, 52, 52)],
            VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch))

