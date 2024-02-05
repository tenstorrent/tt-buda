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
from pybuda._C import DataFormat

class TestPaddingPassC(pybuda.PyBudaModule):

    # Convolutional Network

    def __init__(
        self, 
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.kernel = kernel
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
        self.conv1 = conv2d(name="conv1", kernel=self.kernel, stride=2, in_channels=3)
        self.conv2 = conv2d(name="conv2", kernel=self.kernel, stride=2, in_channels=3)
        self.conv3 = conv2d(name="conv3", kernel=self.kernel, stride=2, in_channels=3)

        # ... head flow 2 ...
        self.conv4 = conv2d(name="conv4", kernel=(5, 5), stride=2, in_channels=3)
        self.conv5 = conv2d(name="conv5", kernel=(5, 5), stride=2, in_channels=3)
        self.conv6 = conv2d(name="conv6", kernel=(5, 5), stride=2, in_channels=3)

        # block
        # ... block flow 1 ...
        self.conv7 = conv2d(name="conv7", kernel=self.kernel, stride=1)
        self.conv8 = conv2d(name="conv8", kernel=self.kernel, stride=1)
        self.conv9 = conv2d(name="conv9", kernel=self.kernel, stride=1)

        # ... block flow 2 ...
        self.conv10 = conv2d(name="conv10", kernel=(5, 5), stride=1)
        self.conv11 = conv2d(name="conv11", kernel=(5, 5), stride=1)
        self.conv12 = conv2d(name="conv12", kernel=(5, 5), stride=1)

        # ... block flow 3 ...
        self.conv13 = conv2d(name="conv13", kernel=(3, 3), stride=1)
        self.conv14 = conv2d(name="conv14", kernel=(3, 3), stride=1)
        self.conv15 = conv2d(name="conv15", kernel=(3, 3), stride=1)

        # intermediate
        # ... intermediate flow 1 ...
        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)

        # ... intermediate flow 2 ...
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)

        # ... intermediate flow 3 ...
        self.maxpool3 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # ... intermediate flow 1, 2 & 3 ...
        self.conv16 = conv2d(name="conv16", kernel=(3, 3), stride=2)
        self.conv17 = conv2d(name="conv17", kernel=(5, 5), stride=2)
        self.conv18 = conv2d(name="conv18", kernel=(11, 11), stride=2)
        self.conv19 = conv2d(name="conv19", kernel=(11, 11), stride=2)
        self.conv20 = conv2d(name="conv20", kernel=(3, 3), stride=2)
        self.conv21 = conv2d(name="conv21", kernel=(5, 5), stride=2)
        self.conv22 = conv2d(name="conv22", kernel=(3, 3), stride=2)
        self.conv23 = conv2d(name="conv23", kernel=(7, 7), stride=2)

        # block
        # ... block flow 4 ...
        self.conv24 = conv2d(name="conv24", kernel=(5, 5), stride=1)
        self.conv25 = conv2d(name="conv25", kernel=(5, 5), stride=1)
        self.conv26 = conv2d(name="conv26", kernel=(5, 5), stride=1)
        self.maxpool4 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # ... block flow 5 ...
        self.conv27 = conv2d(name="conv27", kernel=self.kernel, stride=1)
        self.conv28 = conv2d(name="conv28", kernel=self.kernel, stride=1)
        self.conv29 = conv2d(name="conv29", kernel=self.kernel, stride=1)
        self.maxpool5 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # ... block flow 6 ...
        self.conv30 = conv2d(name="conv30", kernel=(5, 5), stride=1)
        self.conv31 = conv2d(name="conv31", kernel=(5, 5), stride=1)
        self.conv32 = conv2d(name="conv32", kernel=(5, 5), stride=1)
        self.maxpool6 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # ... block flow 7 ...
        self.conv33 = conv2d(name="conv33", kernel=(3, 3), stride=1)
        self.conv34 = conv2d(name="conv34", kernel=(3, 3), stride=1)
        self.conv35 = conv2d(name="conv35", kernel=(3, 3), stride=1)
        self.maxpool7 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # intermediate
        # ... intermediate flow 4 ...
        self.maxpool8 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # ... intermediate flow 5 ...
        self.maxpool9 = maxpool2d(name="maxpool3", kernel=3, stride=2)

        # tail
        # ... tail flow 1 ...
        self.linear1 = linear("lin1")

        # ... tail flow 2 ...
        self.linear2 = linear("lin2")

        # block
        # ... block flow 8 ...
        self.conv36 = conv2d(name="conv36", kernel=(7, 7), stride=1, in_channels=1)
        self.conv37 = conv2d(name="conv37", kernel=(7, 7), stride=1)

    def forward(self, x1, x2):

        # head
        # ... head flow 1 ...
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x1)
        conv3 = self.conv3(x1)
        exp1 = pybuda.op.Exp("exp1", conv1)
        exp2 = pybuda.op.Exp("exp2", conv2)
        exp3 = pybuda.op.Exp("exp3", conv3)

        # ... head flow 2 ...
        conv4 = self.conv4(x2)
        conv5 = self.conv5(x2)
        conv6 = self.conv6(x2)
        exp4 = pybuda.op.Exp("exp4", conv4)
        exp5 = pybuda.op.Exp("exp5", conv5)
        exp6 = pybuda.op.Exp("exp6", conv6)

        # block
        # ... block flow 1 ...
        mm1 = pybuda.op.Matmul("mm1", exp1, exp4)
        conv7 = self.conv7(mm1)
        relu1 = pybuda.op.Relu("relu1", conv7)
        conv8 = self.conv8(relu1)
        relu2 = pybuda.op.Relu("relu2", conv8)
        conv9 = self.conv9(relu2)
        add1 = pybuda.op.Add("add1", mm1, conv9)

        # ... block flow 2 ...
        mm2 = pybuda.op.Matmul("mm2", exp2, exp6)
        conv10 = self.conv10(mm2)
        relu3 = pybuda.op.Relu("relu3", conv10)
        conv11 = self.conv11(relu3)
        exp8 = pybuda.op.Exp("exp8", conv11)
        conv12 = self.conv12(exp8)
        mul1 = pybuda.op.Multiply("mul1", mm2, conv12)

        # ... block flow 3 ...
        mm3 = pybuda.op.Matmul("mm3", exp3, exp5)
        conv13 = self.conv13(mm3)
        exp7 = pybuda.op.Exp("exp7", conv13)
        conv14 = self.conv14(exp7)
        exp9 = pybuda.op.Exp("exp9", conv14)
        conv15 = self.conv15(exp9)
        add2 = pybuda.op.Add("add2", mm3, conv15)

        # intermediate
        # ... intermediate flow 1 ...
        maxpool1 = self.maxpool1(add1)

        # ... intermediate flow 2 ...
        maxpool2 = self.maxpool2(mul1)

        # ... intermediate flow 3 ...
        maxpool3 = self.maxpool3(add2)

        # ... intermediate flow 1, 2 & 3 ...
        mm4 = pybuda.op.Matmul("mm4", maxpool1, maxpool2)
        mm5 = pybuda.op.Matmul("mm5", maxpool2, maxpool3)
        mm6 = pybuda.op.Matmul("mm6", maxpool1, maxpool3)
        exp10 = pybuda.op.Exp("exp10", mm4)
        exp11 = pybuda.op.Exp("exp11", mm6)
        conv16 = self.conv16(mm4)
        conv17 = self.conv17(mm4)
        conv18 = self.conv18(exp10)
        conv19 = self.conv19(mm5)
        conv20 = self.conv20(mm5)
        conv21 = self.conv21(mm6)
        conv22 = self.conv22(mm6)
        conv23 = self.conv23(exp11)
        mul2 = pybuda.op.Multiply("mul2", conv16, conv17)
        add3 = pybuda.op.Add("add3", conv18, conv19)
        add4 = pybuda.op.Add("add4", conv20, conv22)
        mul3 = pybuda.op.Multiply("mul3", conv21, conv23)

        # block
        # ... block flow 4 ...
        maxpool4 = self.maxpool4(mul2)
        conv24 = self.conv24(maxpool4)
        relu4 = pybuda.op.Relu("relu4", conv24)
        conv25 = self.conv25(relu4)
        relu5 = pybuda.op.Relu("relu5", conv25)
        conv26 = self.conv26(relu5)
        add5 = pybuda.op.Add("add5", maxpool4, conv26)

        # ... block flow 5 ...
        maxpool5 = self.maxpool5(add3)
        conv27 = self.conv27(maxpool5)
        relu6 = pybuda.op.Relu("relu6", conv27)
        conv28 = self.conv28(relu6)
        relu7 = pybuda.op.Relu("relu7", conv28)
        conv29 = self.conv29(relu7)
        add6 = pybuda.op.Add("add6", maxpool5, conv29)

        # ... block flow 6 ...
        maxpool6 = self.maxpool6(add4)
        conv30 = self.conv30(maxpool6)
        relu8 = pybuda.op.Relu("relu8", conv30)
        conv31 = self.conv31(relu8)
        relu9 = pybuda.op.Relu("relu9", conv31)
        conv32 = self.conv32(relu9)
        add7 = pybuda.op.Add("add7", maxpool6, conv32)

        # ... block flow 7 ...
        maxpool7 = self.maxpool7(mul3)
        conv33 = self.conv33(maxpool7)
        relu10 = pybuda.op.Relu("relu10", conv33)
        conv34 = self.conv34(relu10)
        relu11 = pybuda.op.Relu("relu11", conv34)
        conv35 = self.conv35(relu11)
        add8 = pybuda.op.Add("add8", maxpool7, conv35)

        # intermediate
        # ... intermediate flow 4 ...
        mm7 = pybuda.op.Matmul("mm7", add5, add8)
        maxpool8 = self.maxpool8(mm7)

        # ... intermediate flow 5 ...
        mm8 = pybuda.op.Matmul("mm8", add6, add7)
        maxpool9 = self.maxpool9(mm8)

        # tail
        # ... tail flow 1 ...
        W1, Z1, R1, C1 = 1, 1, maxpool8.shape[-3], maxpool8.shape[-1] * maxpool8.shape[-2]
        resh1 = pybuda.op.Reshape("resh1", maxpool8, (W1, Z1, R1, C1))
        tr1 = pybuda.op.Transpose("tr1", resh1, -1, -2)
        ra1 = pybuda.op.ReduceAvg("ra1", tr1, -2)
        lin1 = self.linear1(ra1)
        sm1 = pybuda.op.Softmax("sm1", lin1, dim=-1, stable=True)

        # ... tail flow 2 ...
        W2, Z2, R2, C2 = 1, 1, maxpool9.shape[-3], maxpool9.shape[-1] * maxpool9.shape[-2]
        resh2 = pybuda.op.Reshape("resh2", maxpool9, (W2, Z2, R2, C2))
        tr2 = pybuda.op.Transpose("tr2", resh2, -1, -2)
        ra2 = pybuda.op.ReduceAvg("ra2", tr2, -2)
        lin2 = self.linear2(ra2)
        sm2 = pybuda.op.Softmax("sm2", lin2, dim=-1, stable=True)

        # block
        # ... block flow 8 ...
        add9 = pybuda.op.Add("add9", sm1, sm2)
        conv36 = self.conv36(add9)
        relu12 = pybuda.op.Relu("relu12", conv36)
        conv37 = self.conv37(relu12)
        add10 = pybuda.op.Add("add10", add9, conv37)

        return add10



TEST_C_DISABLE_PADDING_PASS_FLAG = True
TEST_C_PRINT_GRAPH_VIZ_FLAG = False
TEST_C_PRINT_GRAPH_AT_FLAG = False
TEST_C_FRACTURING_FLAG = False

TEST_C_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = True
TEST_C_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_C_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_C_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_C_SHAPE_SIZE_1_FLAG = False
TEST_C_SHAPE_SIZE_2_FLAG = False
TEST_C_SHAPE_SIZE_3_FLAG = False

TEST_C_KERNEL_SQUARED_ODD = True
TEST_C_KERNEL_SQUARED_EVEN = False

TEST_C_LOGGER_LEVEL_TRACE = False
TEST_C_LOGGER_LEVEL_DEBUG = False

TEST_C_INPUT_NO = 2


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_C_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    # Environment variable that allows printing a graph
    if TEST_C_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_C_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_C_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_C_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_C_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_C_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_C_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
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

if TEST_C_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_C_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_C_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]


kernel_squared_odd = [
    (1, 1), 
    (3, 3), 
    (5, 5), 
    # (7, 7)
]
kernel_squared_even = [(2, 2), (4, 4), (6, 6)]

kernels = []
if TEST_C_KERNEL_SQUARED_ODD:
    kernels += kernel_squared_odd
if TEST_C_KERNEL_SQUARED_EVEN:
    kernels += kernel_squared_even

kernel_ids = [f"kernel={'x'.join([str(jtem) for jtem in item])}" for item in kernels]

@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.parametrize("in_features", [32])
@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("kernel", kernels, ids=kernel_ids)
def test_padding_pass_c(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    in_features,
    out_features,
    original_shape,
    kernel
):

    pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    set_environment()

    model = TestPaddingPassC(
                name="TestPaddingPassC",
                kernel=kernel,
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
    
    inputs = [Tensor.create_from_torch(torch.rand(act_shape)) for _ in range(TEST_C_INPUT_NO)]

    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *inputs, 
        compiler_cfg=compiler_cfg, 
        verify_cfg=verify_cfg
    )