# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Resnet model cut-outs for testing performance sanity and debug
"""

import numpy as np
import pybuda
import torch

from ..common import benchmark_model
from pybuda.op.common import PyBudaOp
from pybuda.utils import align_up_tile, round_up_div
from pybuda.op.eval.sparse_utils import interleave_tiles, vslice, calculate_conv2d_output_dimensions, create_conv2d_sparse_picker_matrix


class ConvCustomTStreamModule(pybuda.PyBudaModule):
    def __init__(
        self,
        name: str,
        cin: int,
        cout: int,
        kH: int,
        kW: int,
        padding: int,
        stride: int,
        dilation: int,
        conv_mm_t: int,
        sparse_mm_t: int,
    ):
        super().__init__(name)

        self.cin = cin
        self.cout = cout
        self.kH = kH
        self.kW = kW
        self.padding = [padding] * 4
        self.stride = stride
        self.dilation = dilation

        self.conv_mm_t = conv_mm_t
        self.sparse_mm_t = sparse_mm_t

        self.weights = pybuda.Parameter(cout, cin, kH, kW)
        self.weights_pt_tensor = (torch.rand(cout, cin, kH, kW, requires_grad=False, dtype=torch.float32) + 0.00001).detach()
        # self.weights_pt_tensor = (1 + torch.arange(cout * cin * kH * kW, requires_grad=False, dtype=torch.float32).detach().view((cout, cin, kH, kW)))
        self.weights_pt_tensor.requires_grad_(True)
        self.set_parameter("weights", self.weights_pt_tensor)

    def forward(self, x):
        # save original shape
        shape = x.shape
        outy, outx = calculate_conv2d_output_dimensions(shape[-2], shape[-1], (self.kH, self.kW), self.stride, self.padding, self.dilation)

        # activations
        x = pybuda.op.Reshape(f"", x, (1, 1, self.cin, shape[2] * shape[3]))
        x = pybuda.op.Transpose(f"", x, 2, 3)
        x = pybuda.op.PadTile(f"", x, 3, self.cin)
        x = pybuda.op.PadTile(f"", x, 2, shape[2] * shape[3])

        # weights
        w = self.weights
        w = pybuda.op.Reshape("", w, (1, self.cout, self.cin, self.kH * self.kW))
        w = pybuda.op.Transpose("", w, 1, 3, self.kH * self.kW)
        w = pybuda.op.PadTile("", w, 3, self.cout)
        w = pybuda.op.Transpose("", w, 1, 2, self.cin)
        w = pybuda.op.Transpose("", w, -3, -2, self.kH * self.kW)
        w = pybuda.op.HStack("", w, self.kH * self.kW)
        w = pybuda.op.PadTile("", w, 2, self.cin)

        # t stream conv matmul
        if self.conv_mm_t != 1:
            assert x.shape[2] % self.conv_mm_t == 0, "invalid vslice"
            # x = pybuda.op.VSlice("", x, x.shape.rt)
            x = pybuda.op.VSlice("", x, self.conv_mm_t)

        # conv matmul
        x = pybuda.op.Matmul("conv_mm", x, w)

        # maximize t dim
        x = pybuda.op.VSlice("", x, x.shape.rt)

        # Buffer between vslice and hslice
        x = pybuda.op.Buffer("", x)  # HW workaround for: tenstorrent/budabackend#656

        # tms before sparse mm
        x = pybuda.op.HSlice("", x, self.kH * self.kW)
        x = pybuda.op.Buffer("", x)  # HW workaround for: tenstorrent/budabackend#656
        x = pybuda.op.VStack("", x, x.shape[-3] // self.sparse_mm_t)

        # create sparse picker
        pickers = []
        for kY in range(self.kH):
            for kX in range(self.kW):
                y_shift = ((self.kH - 1) // 2) - kY
                x_shift = ((self.kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(shape[2], shape[3], y_shift, x_shift, self.kH, self.kW, [self.stride] * 2, self.padding, self.dilation, tile_align=True)
                pickers.append(picker)

        # Split the sparse tensor
        sparse = interleave_tiles(pickers)  # to match max vslice after conv matmul
        sparse = torch.stack(vslice(sparse, self.sparse_mm_t), dim=0).unsqueeze(0)
        sparse_tensor = pybuda.Tensor.create_from_torch(sparse, constant=True)

        # sparse mm
        x = pybuda.op.SparseMatmul("sparse_mm", sparse_tensor, x)

        # undo t streamed result
        if x.shape.z > 1:
            x = pybuda.op.VStack("", x)

        x = pybuda.op.Buffer("", x)  # vstack can't be followed by other tm ops (transpose below), need to buffer

        # remaining tms to match the regular conv test
        x = pybuda.op.Narrow("", x, 3, 0, self.cout, x.shape[-1])
        x = pybuda.op.Narrow("", x, 2, 0, outy * outx, x.shape[-2])
        x = pybuda.op.Transpose("", x, 2, 3)
        x = pybuda.op.Reshape("", x, (1, self.cout, outy, outx))

        return x

    def forward_golden(self, x):
        return torch.nn.functional.conv2d(
            input=x,
            weight=self.weights_pt_tensor,
            bias=None,
            stride=self.stride,
            padding=(self.kH // 2, self.kW // 2)  # hacky
        )


class ResnetBottleneckReduce(pybuda.PyBudaModule):
    def __init__(
        self,
        name: str,
        ch_lo: int,
        ch_mid: int,
        ch_hi: int,
        use_skip: bool,
        no_reduce: bool = False,
    ):
        super().__init__(name)

        self.use_skip = use_skip
        self.no_reduce = no_reduce

        # left branch
        self.conv_l0 = pybuda.op.nn.Conv2dModule(
            name=name + "_l0",
            in_channels=ch_mid,
            out_channels=ch_hi,
            kernel_size=(1, 1),
            stride=1 if no_reduce else 2,
            padding="same",
            dilation=1,
            groups=1,
            bias=False,
        )

        # right branch
        self.conv_r0 = pybuda.op.nn.Conv2dModule(
            name=name + "_r0",
            in_channels=ch_mid,
            out_channels=ch_lo,
            kernel_size=(1, 1),
            stride=1 if no_reduce else 2,
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

    def forward(self, x):
        r = pybuda.op.Relu(f"", self.conv_r0(x))
        r = pybuda.op.Relu(f"", self.conv_r1(r))
        r = self.conv_r2(r)

        if self.use_skip:
            l = self.conv_l0(x)
            r = pybuda.op.Add(f"", l, r)

        return pybuda.op.Relu(f"", r)


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


class ResnetBlock(pybuda.PyBudaModule):
    def __init__(
        self,
        name: str,
        ch_lo: int,
        ch_mid: int,
        ch_hi: int,
        bottlenecks_cnt: int,  # bottlenecks count
        use_skip: bool,
        use_relu: bool = True,  # This is for debug purposes only, last conv can't have relu if output is directly after
    ):
        super().__init__(name)

        self.red_bn = ResnetBottleneckReduce(
            name=f"{self.name}_btlnck_reduce_0",
            ch_lo=ch_lo,
            ch_mid=ch_mid,
            ch_hi=ch_hi,
            use_skip=use_skip,
        )

        self.bottlenecks = []  # TODO: do we have an equivalent to https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        for idx in range(bottlenecks_cnt - 1):
            # first bn was reduce
            self.bottlenecks.append(
                ResnetBottleneck(
                    name=f"{self.name}_btlnck_{idx + 1}",
                    ch_lo=ch_lo,
                    ch_hi=ch_hi,
                    use_skip=use_skip,
                )
            )

        self.bottlenecks[-1].set_relu(use_relu)

    def forward(self, x):
        y = self.red_bn(x)

        for bn in self.bottlenecks:
            y = bn(y)

        return y


class ResnetBlock2(pybuda.PyBudaModule):
    def __init__(
        self,
        name: str,
        ch_lo: int,
        ch_hi: int,
        use_skip: bool,
    ):
        super().__init__(name)

        # The first bottleneck in block 2 is unique, it's the only one with a conv on left side, but no reduce
        self.bottleneck0 = ResnetBottleneckReduce(
            name=f"{self.name}_btlnck_reduce_0",
            ch_lo=ch_lo,
            ch_mid=ch_lo,  # keep correct num of channels
            ch_hi=ch_hi,
            use_skip=use_skip,
            no_reduce=True,  # change strides of 2 to 1 in order to avoid reduce
        )

        self.bottleneck1 = ResnetBottleneck(
            name=f"{self.name}_btlnck_1",
            ch_lo=ch_lo,
            ch_hi=ch_hi,
            use_skip=use_skip,
        )

        self.bottleneck2 = ResnetBottleneck(
            name=f"{self.name}_btlnck_2",
            ch_lo=ch_lo,
            ch_hi=ch_hi,
            use_skip=use_skip,
        )

    def forward(self, x):
        # TODO: Missing beginning
        y = self.bottleneck0(x)
        y = self.bottleneck1(y)
        y = self.bottleneck2(y)

        return y


class Resnet(pybuda.PyBudaModule):
    def __init__(
        self,
        name: str,
        use_skip: bool,
    ):
        super().__init__(name)

        self.big_conv = ConvCustomTStreamModule(
            name="",
            cin=3,
            cout=64,
            kH=7,
            kW=7,
            padding=3,
            stride=2,
            dilation=1,
            conv_mm_t=1568,
            sparse_mm_t=14,
        )

        self.big_conv_automatic = pybuda.op.nn.Conv2dModule(
            name=name + "_big_conv_automatic",
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=2,
            padding=3,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.max_pool = pybuda.op.nn.MaxPool2dModule(
            name=name + "_max_pool",
            kernel_size=3,
            stride=2,
        )

        self.block2 = ResnetBlock2(
            name=name + "_block2",
            ch_lo=64,
            ch_hi=256,
            use_skip=use_skip,
        )

        self.block3 = ResnetBlock(
            name=name + "_block3",
            ch_lo=128,
            ch_mid=256,
            ch_hi=512,
            bottlenecks_cnt=4,
            use_skip=use_skip,
        )

        self.block4 = ResnetBlock(
            name=name + "_block4",
            ch_lo=256,
            ch_mid=512,
            ch_hi=1024,
            bottlenecks_cnt=6,
            use_skip=use_skip,
        )

        self.block5 = ResnetBlock(
            name=name + "_block5",
            ch_lo=512,
            ch_mid=1024,
            ch_hi=2048,
            bottlenecks_cnt=3,
            use_skip=use_skip,
            use_relu=False,
        )

        self.linear = pybuda.op.nn.Linear(
            name=name + "_linear",
            in_features=2048,
            out_features=1000,
            bias=False,
        )


    def forward(self, x):
        # TODO: add batchnorms

        # # # Head: 224x224 conv, relu, maxpool
        # y = pybuda.op.Relu(self.name + "_big_conv_relu", self.big_conv(x))
        # y = self.max_pool(y)

        # # Just maxpool
        # y = self.max_pool(x)  # <---- x

        # Auto conv
        y = pybuda.op.Relu(self.name + "_big_conv_relu", self.big_conv_automatic(x))
        y = self.max_pool(y)

        # 4 blocks of bottlenecks of convs (no batchnorm)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)

        # Tail: global avg pool + linear + softmax
        y = pybuda.op.Reshape("", y, (1, 1, y.shape[-3], y.shape[-1] * y.shape[-2]))
        y = pybuda.op.Transpose("", y, -1, -2)
        y = pybuda.op.ReduceAvg("", y, -2)
        y = self.linear(y)
        y = pybuda.op.Softmax("", y, dim=-1, stable=True)

        return y

@benchmark_model(configs=["resnet", "resnet_no_skip_conns"])
def resnet(training: bool, config: str, microbatch: int, devtype: str, arch: str):
    input_size = (224, 224)
    cin = 3

    if config == "resnet":
        use_skip = True
    elif config == "resnet_no_skip_conns":
        use_skip = False
    else:
        raise RuntimeError(f"Invalid config: {config}")

    if microbatch == 0:
        microbatch = 1

    models = {"tt": Resnet(config, use_skip)}
    inputs = [torch.rand(1, cin, input_size[0], input_size[1])]
    targets = []

    if training:
        assert False
        # models["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return models, inputs, targets, {}
