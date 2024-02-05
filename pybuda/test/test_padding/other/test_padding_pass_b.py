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


class TestPaddingPassB(pybuda.PyBudaModule):

    # Convolutional Network

    def __init__(
        self, 
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features,
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

        # Note: kernel size depends on convolution

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
        # self.conv3 = conv2d(name="conv3", kernel=(7, 7), stride=2, in_channels=3)
        self.conv3 = conv2d(name="conv3", kernel=(5, 5), stride=2, in_channels=3)

        # ... head flow 2 ...
        self.conv4 = conv2d(name="conv4", kernel=(5, 5), stride=2, in_channels=3)
        self.conv5 = conv2d(name="conv5", kernel=(5, 5), stride=2, in_channels=3)
        self.conv6 = conv2d(name="conv6", kernel=(5, 5), stride=2, in_channels=3)
        self.conv7 = conv2d(name="conv7", kernel=(5, 5), stride=2, in_channels=3)

        # ... head flow 1 & 2 ...
        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)
        self.maxpool3 = maxpool2d(name="maxpool3", kernel=3, stride=2)
        self.maxpool4 = maxpool2d(name="maxpool4", kernel=3, stride=2)
        self.maxpool5 = maxpool2d(name="maxpool5", kernel=3, stride=2)

        # block 
        # ... block flow 1 ...
        self.conv8 = conv2d(name="conv8", kernel=(3, 3), stride=1)
        self.conv9 = conv2d(name="conv9", kernel=(3, 3), stride=1)
        self.conv10 = conv2d(name="conv10", kernel=(3, 3), stride=1)

        # ... block flow 2 ...
        self.conv11 = conv2d(name="conv11", kernel=(5, 5), stride=1)
        self.conv12 = conv2d(name="conv12", kernel=(5, 5), stride=1)
        self.conv13 = conv2d(name="conv13", kernel=(5, 5), stride=1)

        # ... block flow 3 ...
        self.conv14 = conv2d(name="conv14", kernel=(3, 3), stride=1)
        self.conv15 = conv2d(name="conv15", kernel=(3, 3), stride=1)
        self.conv16 = conv2d(name="conv16", kernel=(3, 3), stride=1)

        # ... block flow 4 ...
        self.conv17 = conv2d(name="conv17", kernel=(11, 11), stride=1)
        self.conv18 = conv2d(name="conv18", kernel=(11, 11), stride=1)
        self.conv19 = conv2d(name="conv19", kernel=(11, 11), stride=1)

        # ... block flow 5 ...
        self.conv20 = conv2d(name="conv20", kernel=(5, 5), stride=1)
        self.conv21 = conv2d(name="conv21", kernel=(5, 5), stride=1)
        self.conv22 = conv2d(name="conv22", kernel=(5, 5), stride=1)

        # tail
        self.linear1 = linear(name="lin1")
        self.linear2 = linear(name="lin2")

    def forward(self, x1, x2):
        
        # head
        # ... head flow 1 ...
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x1)
        conv3 = self.conv3(x1)

        relu1 = pybuda.op.Relu("relu1", conv1)
        relu2 = pybuda.op.Relu("relu2", conv3)

        # ... head flow 2 ...
        conv4 = self.conv4(x2)
        conv5 = self.conv5(x2)
        conv6 = self.conv6(x2)
        conv7 = self.conv7(x2)

        relu3 = pybuda.op.Relu("relu3", conv5)
        relu4 = pybuda.op.Relu("relu4", conv7)

        # ... head flow 1 & 2 ...
        mul1 = pybuda.op.Multiply("mul1", conv2, conv4)
        add1 = pybuda.op.Add("add1", relu2, conv6)
        mul2 = pybuda.op.Multiply("mul2", relu3, relu4)

        maxpool1 = self.maxpool1(relu1)
        maxpool2 = self.maxpool2(mul1)
        maxpool3 = self.maxpool3(add1)
        maxpool4 = self.maxpool4(conv4)
        maxpool5 = self.maxpool5(mul2)

        # block
        # ... block flow 1 ...
        add2 = pybuda.op.Add("add2", maxpool1, maxpool2)
        relu5 = pybuda.op.Relu("relu5", add2)
        conv8 = self.conv8(relu5)
        relu6 = pybuda.op.Relu("relu6", conv8)
        conv9 = self.conv9(relu6)
        relu7 = pybuda.op.Relu("relu7", conv9)
        conv10 = self.conv10(relu7)
        add5 = pybuda.op.Add("add5", relu5, conv10)

        # ... block flow 2 ...
        add3 = pybuda.op.Add("add3", maxpool3, maxpool4)
        mul3 = pybuda.op.Multiply("mul3", relu5, add3)
        exp1 = pybuda.op.Exp("exp1", mul3)

        conv11 = self.conv11(add3)
        relu8 = pybuda.op.Relu("relu8", conv11)
        exp2 = pybuda.op.Exp("exp2", relu8)
        conv12 = self.conv12(exp2)
        conv13 = self.conv13(conv12)
        exp3 = pybuda.op.Exp("exp3", conv13)
        add6 = pybuda.op.Add("add6", exp1, exp3)

        # ... block flow 3 ...
        add4 = pybuda.op.Add("add4", maxpool4, maxpool5)
        conv14 = self.conv14(add4)
        relu9 = pybuda.op.Relu("relu9", conv14)
        conv15 = self.conv15(relu9)
        exp4 = pybuda.op.Exp("exp4", conv15)
        conv16 = self.conv16(exp4)
        mul4 = pybuda.op.Multiply("mul4", add4, conv16)

        # ... block flow 4 ...
        mm1 = pybuda.op.Matmul("mm1", add5, add6)  # maybe add operation, add7
        relu10 = pybuda.op.Relu("relu10", mm1)
        conv17 = self.conv17(relu10)
        relu11 = pybuda.op.Relu("relu11", conv17)
        conv18 = self.conv18(relu11)
        relu12 = pybuda.op.Relu("relu12", conv18)
        conv19 = self.conv19(relu12)
        add7 = pybuda.op.Add("add7", relu10, conv19)

        # ... block flow 5 ...
        relu13 = pybuda.op.Relu("relu13", mul4)
        conv20 = self.conv20(relu13)
        relu14 = pybuda.op.Relu("relu14", conv20)
        conv21 = self.conv21(relu14)
        relu15 = pybuda.op.Relu("relu15", conv21)
        conv22 = self.conv22(relu15)
        mm2 = pybuda.op.Matmul("mm2", relu10, relu13)
        add8 = pybuda.op.Add("add8", mm2, conv22)

        # tail
        # ... tail flow 1 ...
        W1, Z1, R1, C1 = 1, 1, add7.shape[-3], add7.shape[-1] * add7.shape[-2]
        resh1 = pybuda.op.Reshape("resh1", add7, (W1, Z1, R1, C1))
        tr1 = pybuda.op.Transpose("tr1", resh1, -1, -2)
        ra1 = pybuda.op.ReduceAvg("ra1", tr1, -2)
        lin1 = self.linear1(ra1)
        sm1 = pybuda.op.Softmax("sm1", lin1, dim=-1, stable=True)

        # ... tail flow 2 ...
        W2, Z2, R2, C2 = 1, 1, add8.shape[-3], add8.shape[-1] * add8.shape[-2]
        resh2 = pybuda.op.Reshape("resh2", add8, (W2, Z2, R2, C2))
        tr2 = pybuda.op.Transpose("tr2", resh2, -1, -2)
        ra2 = pybuda.op.ReduceAvg("ra2", tr2, -2)
        lin2 = self.linear2(ra2)
        sm2 = pybuda.op.Softmax("sm2", lin2, dim=-1, stable=True)

        return sm1, sm2


class TestPaddingPassB_1(pybuda.PyBudaModule):

    def __init__(
        self, 
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features,
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

        # Note: kernel size depends on convolution

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

        # self.conv1 = conv2d(name="conv1", kernel=(7, 7), stride=2, in_channels=3)
        self.conv1 = conv2d(name="conv1", kernel=(5, 5), stride=2, in_channels=3)
        # self.conv2 = conv2d(name="conv2", kernel=(7, 7), stride=2, in_channels=3)
        self.conv2 = conv2d(name="conv2", kernel=(5, 5), stride=2, in_channels=3)

        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)

    def forward(self, x1, x2):
        
        # Layer 2
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)

        maxpool1 = self.maxpool1(conv1)
        maxpool2 = self.maxpool2(conv2)

        # Layer 3
        add = pybuda.op.Add("add", maxpool1, maxpool2)

        return add


class TestPaddingPassB_2(pybuda.PyBudaModule):
    
    def __init__(
        self, 
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features,
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

        # Note: kernel size depends on convolution

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

        # self.conv1 = conv2d(name="conv1", kernel=(7, 7), stride=2, in_channels=3)
        self.conv1 = conv2d(name="conv1", kernel=(5, 5), stride=2, in_channels=3)
        # self.conv2 = conv2d(name="conv2", kernel=(7, 7), stride=2, in_channels=3)
        self.conv2 = conv2d(name="conv2", kernel=(5, 5), stride=2, in_channels=3)

        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)

    def forward(self, x1, x2):

        # Layer 2
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)

        # Layer 3
        add = pybuda.op.Add("add", conv1, conv2)
        mul = pybuda.op.Multiply("mul", conv1, conv2)

        # Layer 4
        maxpool1 = self.maxpool1(add)
        maxpool2 = self.maxpool2(mul)

        # Layer 5
        sub = pybuda.op.Subtract("sub", maxpool1, maxpool2)

        return sub

class TestPaddingPassB_3(pybuda.PyBudaModule):

    def __init__(
        self, 
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features,
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

        # Note: kernel size depends on convolution

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

        # self.conv1 = conv2d(name="conv1", kernel=(7, 7), stride=2, in_channels=3)
        self.conv1 = conv2d(name="conv1", kernel=(5, 5), stride=2, in_channels=3)

        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)

    def forward(self, x1, x2):

        # Layer 2
        sub = pybuda.op.Subtract("sub", x1, x2)
        # Layer 3
        conv = self.conv1(sub)
        # Layer 4
        add = pybuda.op.Add("add", conv, conv)
        # Layer 5
        maxpool = self.maxpool1(add)
        # Layer 6
        exp = pybuda.op.Exp("exp", maxpool)

        return exp


TEST_B_DISABLE_PADDING_PASS_FLAG = True
TEST_B_PRINT_GRAPH_VIZ_FLAG = False
TEST_B_PRINT_GRAPH_AT_FLAG = False
TEST_B_FRACTURING_FLAG = False

TEST_B_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = False
TEST_B_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = False
TEST_B_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = False
TEST_B_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_B_SHAPE_SIZE_1_FLAG = False
TEST_B_SHAPE_SIZE_2_FLAG = False
TEST_B_SHAPE_SIZE_3_FLAG = False

TEST_B_KERNEL_SQUARED_ODD = True
TEST_B_KERNEL_SQUARED_EVEN = False

TEST_B_MODEL_B_FLAG = False
TEST_B_MODEL_B_1_FLAG = False
TEST_B_MODEL_B_2_FLAG = False
TEST_B_MODEL_B_3_FLAG = False

TEST_B_LOGGER_LEVEL_TRACE = False
TEST_B_LOGGER_LEVEL_DEBUG = False

TEST_B_INPUT_NO = 2


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_B_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    # Environment variable that allows printing a graph
    if TEST_B_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_B_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_B_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_B_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_B_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_B_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_B_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
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
if TEST_B_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_B_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_B_SHAPE_SIZE_3_FLAG:
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
if TEST_B_KERNEL_SQUARED_EVEN:
    kernels += kernel_squared_even
if TEST_B_KERNEL_SQUARED_ODD:
    kernels += kernel_squared_odd

kernel_ids = [f"kernel={'x'.join([str(jtem) for jtem in item])}" for item in kernels]

test_model = []
if TEST_B_MODEL_B_FLAG:
    test_model.append(("TestPaddingPassB", TestPaddingPassB))
if TEST_B_MODEL_B_1_FLAG:
    test_model.append(("TestPaddingPassB_1", TestPaddingPassB_1))
if TEST_B_MODEL_B_2_FLAG:
    test_model.append(("TestPaddingPassB_2", TestPaddingPassB_2))
if TEST_B_MODEL_B_3_FLAG:
    test_model.append(("TestPaddingPassB_3", TestPaddingPassB_3))


@pytest.mark.parametrize("test_model", test_model, ids=[item[0] for item in test_model])
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.parametrize("in_features", [32])
@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("kernel", kernels, ids=kernel_ids)
def test_padding_pass_b(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    in_features,
    out_features,
    original_shape,
    kernel,
    test_model
):

    if test_kind.is_training():
        pytest.skip()

    set_environment()

    model = test_model[1](
        name=test_model[0],
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
    
    inputs = [Tensor.create_from_torch(torch.rand(act_shape)) for _ in range(TEST_B_INPUT_NO)]

    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *inputs, 
        compiler_cfg=compiler_cfg, 
        verify_cfg=verify_cfg
    )