# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Padding Pass Tests
#
import os

import torch
import pytest

import pybuda
from pybuda import (
    TTDevice,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
)
from pybuda._C.backend_api import BackendType


class TestPaddingPassD(pybuda.PyBudaModule):

    # Convolutional Network

    def __init__(
        self, 
        name,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed for particular test
        self.padding = "same"
        self.dilation = 1
        self.groups = 1
        self.bias = False

        # Note: kernel and stride depend on convolution

        # Auxiliary function that creates convolutional layer
        def conv2d(name, kernel, stride, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # Auxiliary function that creates maxpool layer
        def maxpool2d(name, kernel, stride):
            return pybuda.op.nn.MaxPool2dModule(
                        name=f"{self.name}.{name}",
                        kernel_size=kernel,
                        stride=stride,
                    )

        # Auxiliary function that creates FC layer(Linear) layer
        # We use this layer at the end of the CNN before Softmax
        def linear(name):
            return pybuda.op.nn.Linear(
                        name=f"{self.name}.{name}",
                        in_features=self.in_features,
                        out_features=self.out_features,
                        bias=False,
                    )

        # head
        # ... head flow 1 ...
        # self.conv1 = conv2d(name="conv1", kernel=(7, 7), stride=2, in_channels=3)
        self.conv1 = conv2d(name="conv1", kernel=(5, 5), stride=2, in_channels=3)
        # self.conv2 = conv2d(name="conv2", kernel=(7, 7), stride=2, in_channels=3)
        self.conv2 = conv2d(name="conv2", kernel=(5, 5), stride=2, in_channels=3)

        # ... head flow 2 ...
        # self.conv3 = conv2d(name="conv3", kernel=(7, 7), stride=2, in_channels=3)
        self.conv3 = conv2d(name="conv3", kernel=(5, 5), stride=2, in_channels=3)
        # self.conv4 = conv2d(name="conv4", kernel=(7, 7), stride=2, in_channels=3)
        self.conv4 = conv2d(name="conv4", kernel=(5, 5), stride=2, in_channels=3)

        # ... head flow 3 ...
        # self.conv5 = conv2d(name="conv5", kernel=(7, 7), stride=2, in_channels=3)
        self.conv5 = conv2d(name="conv5", kernel=(5, 5), stride=2, in_channels=3)
        # self.conv6 = conv2d(name="conv6", kernel=(7, 7), stride=2, in_channels=3)
        self.conv6 = conv2d(name="conv6", kernel=(5, 5), stride=2, in_channels=3)

        # block
        # ... block flow 1 ...
        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)
        self.conv7 = conv2d(name="conv7", kernel=(5, 5), stride=1)
        self.conv8 = conv2d(name="conv8", kernel=(5, 5), stride=1)
        self.conv9 = conv2d(name="conv9", kernel=(5, 5), stride=1)

        # ... block flow 2 ...
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)
        self.conv12 = conv2d(name="conv12", kernel=(5, 5), stride=1)
        self.conv13 = conv2d(name="conv13", kernel=(5, 5), stride=1)
        self.conv14 = conv2d(name="conv14", kernel=(5, 5), stride=1)

        # ... block flow 3 ...
        self.maxpool3 = maxpool2d(name="maxpool3", kernel=3, stride=2)
        self.conv17 = conv2d(name="conv17", kernel=(5, 5), stride=1)
        self.conv18 = conv2d(name="conv18", kernel=(5, 5), stride=1)
        self.conv19 = conv2d(name="conv19", kernel=(5, 5), stride=1)

        # ... block flow 1 & 2 ...
        self.conv10 = conv2d(name="conv10", kernel=(5, 5), stride=1)
        self.conv11 = conv2d(name="conv11", kernel=(5, 5), stride=1)

        # ... block flow 2 & 3 ...
        self.conv15 = conv2d(name="conv15", kernel=(5, 5), stride=1)
        self.conv16 = conv2d(name="conv16", kernel=(5, 5), stride=1)

        # ... block flow 4 ...
        self.conv20 = conv2d(name="conv20", kernel=(3, 3), stride=1)
        self.conv21 = conv2d(name="conv21", kernel=(3, 3), stride=1)
        self.conv22 = conv2d(name="conv22", kernel=(3, 3), stride=1)
        
        # ... block flow 5 ...
        self.conv23 = conv2d(name="conv23", kernel=(3, 3), stride=1)
        self.conv24 = conv2d(name="conv24", kernel=(3, 3), stride=1)
        self.conv25 = conv2d(name="conv25", kernel=(3, 3), stride=1)

        # ... block flow 6 ...
        self.conv26 = conv2d(name="conv26", kernel=(3, 3), stride=1)
        self.conv27 = conv2d(name="conv27", kernel=(3, 3), stride=1)
        self.conv28 = conv2d(name="conv28", kernel=(3, 3), stride=1)

        # tail
        # ... tail flow 1 ...
        self.linear1 = linear("lin1")

        # ... tail flow 2 ...
        self.linear2 = linear("lin2")

    def forward(self, x1, x2, x3):

        # head
        # ... head flow 1 ...
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x1)
        mm1 = pybuda.op.Matmul("mm1", conv1, conv2) 

        # ... head flow 2 ...
        conv3 = self.conv3(x2)
        conv4 = self.conv4(x2)
        mm2 = pybuda.op.Matmul("mm2", conv3, conv4)

        # ... head flow 3 ...
        conv5 = self.conv5(x3)
        conv6 = self.conv6(x3)
        mm3 = pybuda.op.Matmul("mm3", conv5, conv6)

        # block
        # ... block flow 1 ...
        maxpool1 = self.maxpool1(mm1)
        conv7 = self.conv7(maxpool1)
        exp1 = pybuda.op.Exp("exp1", conv7)
        conv8 = self.conv8(exp1)
        exp2 = pybuda.op.Exp("exp2", conv8)
        conv9 = self.conv9(exp2)
        add13 = pybuda.op.Add("add13", maxpool1, conv9)

        # ... block flow 2 ...
        maxpool2 = self.maxpool2(mm2)
        conv12 = self.conv12(maxpool2)
        exp3 = pybuda.op.Exp("exp3", conv12)
        conv13 = self.conv13(exp3)
        exp4 = pybuda.op.Exp("exp4", conv13)
        conv14 = self.conv14(exp4)
        add4 = pybuda.op.Add("add4", maxpool2, conv14)

        # ... block flow 3 ...
        maxpool3 = self.maxpool3(mm3)
        conv17 = self.conv17(maxpool3)
        exp5 = pybuda.op.Exp("exp5", conv17)
        conv18 = self.conv18(exp5)
        exp6 = pybuda.op.Exp("exp6", conv18)
        conv19 = self.conv19(exp6)
        add6 = pybuda.op.Add("add6", maxpool3, conv19)

        # ... block flow 1 & 2 ...
        mul1 = pybuda.op.Multiply("mul1", conv7, conv12)
        sm1 = pybuda.op.Softmax("sm1", mul1, dim=-2, stable=True)
        conv10 = self.conv10(sm1)
        add2_ = pybuda.op.Add("add2_", conv8, conv13)
        add2 = pybuda.op.Add("add2", add2_, conv10)
        conv11 = self.conv11(add2)
        mul2_ = pybuda.op.Multiply("mul2_", conv9, conv14)
        mul2 = pybuda.op.Multiply("mul2", mul2_, conv11)

        # ... block flow 2 & 3 ...
        add1 = pybuda.op.Add("add1", conv12, conv17)
        sm2 = pybuda.op.Softmax("sm2", add1, dim=-2, stable=True)
        conv15 = self.conv15(sm2)
        add3_ = pybuda.op.Add("add3_", conv15, conv18)
        add3 = pybuda.op.Add("add3", add3_, conv13)
        conv16 = self.conv16(add3)
        add5_ = pybuda.op.Add("add5_", conv14, conv16)
        add5 = pybuda.op.Add("add5", add5_, conv19)

        # ... block flow 4 ...
        add7 = pybuda.op.Add("add7", add13, mul2)
        conv20 = self.conv20(add7)
        relu1 = pybuda.op.Relu("relu1", conv20)
        conv21 = self.conv21(relu1)
        relu2 = pybuda.op.Relu("relu2", conv21)
        conv22 = self.conv22(relu2)
        add10 = pybuda.op.Add("add10", add7, conv22)
        exp7 = pybuda.op.Exp("exp7", add10)
        red1 = pybuda.op.ReduceSum("red1", exp7, dim=-2)

        # ... block flow 5 ...
        add8 = pybuda.op.Add("add8", add4, add5)
        conv23 = self.conv23(add8)
        relu3 = pybuda.op.Relu("relu3", conv23)
        conv24 = self.conv24(relu3)
        relu4 = pybuda.op.Relu("relu4", conv24)
        conv25 = self.conv25(relu4)
        add11 = pybuda.op.Add("add11", add8, conv25)
        exp8 = pybuda.op.Exp("exp8", add11)
        red2 = pybuda.op.ReduceSum("red2", exp8, dim=-2)

        # ... block flow 6 ...
        add9 = pybuda.op.Add("add9", add5, add6)
        conv26 = self.conv26(add9)
        relu5 = pybuda.op.Relu("relu5", conv26)
        conv27 = self.conv27(relu5)
        relu6 = pybuda.op.Relu("relu6", conv27)
        conv28 = self.conv28(relu6)
        add12 = pybuda.op.Add("add12", add9, conv28)
        exp9 = pybuda.op.Exp("exp9", add12)
        red3 = pybuda.op.ReduceSum("red3", exp9, dim=-2)

        # tail
        # ... tail flow 1 ...
        add15 = pybuda.op.Add("add15", red1, red2)
        W1, Z1, R1, C1 = 1, 1, add15.shape[-3], add15.shape[-1] * add15.shape[-2]
        resh1 = pybuda.op.Reshape("resh1", add15, (W1, Z1, R1, C1))
        tr1 = pybuda.op.Transpose("tr1", resh1, -2, -1)
        ra1 = pybuda.op.ReduceAvg("ra1", tr1, -2)
        lin1 = self.linear1(ra1)
        sm3 = pybuda.op.Softmax("sm3", lin1, dim=-1, stable=True)

        # ... tail flow 2 ...
        add14 = pybuda.op.Add("add14", red2, red3)
        W2, Z2, R2, C2 = 1, 1, add14.shape[-3], add14.shape[-1] * add14.shape[-2]
        resh2 = pybuda.op.Reshape("resh2", add14, (W2, Z2, R2, C2))
        tr2 = pybuda.op.Transpose("tr2", resh2, -1, -2)
        ra2 = pybuda.op.ReduceAvg("ra2", tr2, -2)
        lin2 = self.linear2(ra2)
        sm4 = pybuda.op.Softmax("sm4", lin2, dim=-1, stable=True)

        return sm3, sm4


TEST_D_DISABLE_PADDING_PASS_FLAG = True
TEST_D_PRINT_GRAPH_VIZ_FLAG = False
TEST_D_PRINT_GRAPH_AT_FLAG = False
TEST_D_FRACTURING_FLAG = False

TEST_D_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = True
TEST_D_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_D_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_D_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_D_SHAPE_SIZE_1_FLAG = False
TEST_D_SHAPE_SIZE_2_FLAG = False
TEST_D_SHAPE_SIZE_3_FLAG = False

TEST_D_KERNEL_SQUARED_ODD = True
TEST_D_KERNEL_SQUARED_EVEN = True

TEST_D_LOGGER_LEVEL_TRACE = False
TEST_D_LOGGER_LEVEL_DEBUG = False

TEST_D_INPUT_NO = 3


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_D_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    # Environment variable that allows printing a graph
    if TEST_D_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_D_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_D_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_D_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_D_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_D_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_D_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
        os.environ["PYBUDA_LEGALIZER_DEBUG_NODE_NAME"] = ""

# The main reason why we use shapes of different sizes is
# because convolutional neural networks can't always work with big shapes.

# Each size category is bigger than previous one 
# for one or half order of magnitude

# Shapes -> Size 1
shapes_size_1 = [
    (32, 32),
    (37, 37),
    (64, 64),
    (73, 73)
]

# Shapes -> Size 2
shapes_size_2 = [
    (128, 128), 
    (192, 192), 
    (211, 211), 
    (212, 212)
]

# Shapes -> Size 3
shapes_size_3 = [
    (541, 541),
    (977, 977),
    (768, 768),
    (1024, 1024)
]

original_shape = []

if TEST_D_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_D_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_D_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]

@pytest.mark.parametrize("original_shape", original_shape)
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.parametrize("in_features", [32])
@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("in_channels", [32])
def test_padding_pass_d(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    in_features,
    out_features,
    original_shape
):

    pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    set_environment()

    model = TestPaddingPassD(
                name="TestPaddingPassD",
                in_channels=in_channels,
                out_channels=out_channels,
                in_features=in_features,
                out_features=out_features
            )

    if test_kind.is_training() and test_device.devtype == BackendType.Silicon:
        relative_atol = 0.3
    else: 
        relative_atol = 0.1
        
    if test_device.devtype == BackendType.Silicon:
        pcc = 0.96
    else:
        pcc = 0.99

    # activations = ((torch.rand((1, 3, original_shape[0], original_shape[1])) + 0.00001) * 100000).detach()
    act_shape = (1, in_channels, original_shape[0], original_shape[1])

    compiler_cfg = CompilerConfig(
        enable_training=test_kind.is_training()
    )
    verify_cfg = VerifyConfig(
        pcc=pcc,
        relative_atol=relative_atol
    )
    
    inputs = [Tensor.create_from_torch(torch.rand(act_shape)) for _ in range(TEST_D_INPUT_NO)]

    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *inputs, 
        compiler_cfg=compiler_cfg, 
        verify_cfg=verify_cfg
    )