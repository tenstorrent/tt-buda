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

class TestPaddingPassF(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is combination of convolutional layers,
    # element-wise operations and resnet blocks.

    def __init__(
        self,
        name,
        shape,
        in_channels,
        out_channels,
        in_features,
        out_features,
        kernel
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        
        # Fixed for particular test
        self.padding = "same"
        self.dilation = 1
        self.groups = 1
        self.bias = False
        self.stride = 1

        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(5)]
        for i in range(1, 6):
            self.set_parameter(
                "train_param" + str(i), 
                torch.rand(*self.shape, requires_grad=True)
            )

        # Auxiliary function that creates convolutional layer
        def conv2d(name, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel,
                        stride=self.stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # Auxiliary function that creates maxpool layer
        def maxpool2d(name, kernel=2, stride=1):
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

        # ... head ...

        # ... block ...
        #   ... block flow 1 ...
        self.conv1 = conv2d(name="conv1")
        self.conv2 = conv2d(name="conv2")
        self.conv3 = conv2d(name="conv3")

        self.conv4 = conv2d(name="conv4")

        #   ... block flow 2 ...
        self.conv5 = conv2d(name="conv5")
        self.conv6 = conv2d(name="conv6")
        self.conv7 = conv2d(name="conv7")

        #   ... block flow 3 ...
        self.conv8 = conv2d(name="conv8")
        self.conv9 = conv2d(name="conv9")
        self.conv10 = conv2d(name="conv10")

        # ... intermediate ...
        self.maxpool1 = maxpool2d("pool1")
        self.maxpool2 = maxpool2d("pool2")

        # ... block ...
        self.conv11 = conv2d(name="conv11")
        self.conv12 = conv2d(name="conv12")
        self.conv13 = conv2d(name="conv13")
        self.conv14 = conv2d(name="conv14")
        self.conv15 = conv2d(name="conv15")
        self.conv16 = conv2d(name="conv16")

        # ... tail ...
        self.lin1 = linear("lin1")
        self.lin2 = linear("lin2")
        self.lin3 = linear("lin3")

    def forward(self, x1, x2, x3):
        
        # ... head ...
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x1, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)
        mul4 = pybuda.op.Multiply("mul4", x3, self.train_param2)
        mul5 = pybuda.op.Multiply("mul5", x3, self.train_param3)

        # ... block ...
        add1 = pybuda.op.Add("add1", mul4, mul5)
        conv4 = self.conv4(mul3)
        mul6 = pybuda.op.Multiply("mul6", mul2, conv4)
        mul7 = pybuda.op.Multiply("mul7", mul3, add1)
        add2 = pybuda.op.Add("add2", mul3, mul6)

        #   ... block flow 1 ...
        conv1 = self.conv1(mul1)
        relu1 = pybuda.op.Relu("relu1", conv1)
        conv2 = self.conv2(relu1)
        relu2 = pybuda.op.Relu("relu2", conv2) 
        conv3 = self.conv3(relu2)
        add3 = pybuda.op.Add("add3", mul1, conv3)
        relu7 = pybuda.op.Relu("relu7", add3)
        mul9 = pybuda.op.Multiply("mul9", relu7, mul7)

        #   ... block flow 2 ...
        conv5 = self.conv5(mul6)
        relu3 = pybuda.op.Relu("relu3", conv5)
        conv6 = self.conv6(relu3)
        relu4 = pybuda.op.Relu("relu4", conv6)
        conv7 = self.conv7(relu4)
        add4 = pybuda.op.Add("add4", add2, conv7)
        relu8 = pybuda.op.Relu("relu8", add4)
        add5 = pybuda.op.Add("add5", conv7, mul7)
        relu9 = pybuda.op.Relu("relu9", add5)

        #   ... block flow 3 ...
        conv8 = self.conv8(mul5)
        relu5 = pybuda.op.Relu("relu5", conv8)
        conv9 = self.conv9(relu5)
        relu6 = pybuda.op.Relu("relu6", conv9)
        conv10 = self.conv10(relu6)
        mul8 = pybuda.op.Multiply("mul8", mul5, conv10)
        exp1 = pybuda.op.Exp("exp1", mul8)
        add6 = pybuda.op.Add("add6", exp1, mul7)
        mul10 = pybuda.op.Multiply("mul10", add5, exp1)
        relu10 = pybuda.op.Relu("relu10", mul10)

        # ... intermediate ...
        add7 = pybuda.op.Add("add7", mul9, relu10)
        add8 = pybuda.op.Add("add8", mul2, relu9)
        add9 = pybuda.op.Add("add9", relu8, relu10)
        add10 = pybuda.op.Add("add10", relu8, add6)
        maxpool1 = self.maxpool1(add7)
        maxpool2 = self.maxpool2(add10)
        mul11 = pybuda.op.Multiply("mul11", add8, add9)
        mul12 = pybuda.op.Multiply("mul12", add7, add10)
        mul13 = pybuda.op.Multiply("mul13", maxpool1, maxpool2)

        # ... block ...
        #   ... block flow 4 ...
        conv11 = self.conv11(mul11)
        relu11 = pybuda.op.Relu("relu11", conv11)
        conv12 = self.conv12(relu11)
        relu12 = pybuda.op.Relu("relu12", conv12)
        conv13 = self.conv13(relu12)
        add11 = pybuda.op.Add("add11", conv13, mul11)

        #   ... block flow 5 ...
        conv14 = self.conv14(mul12)
        relu13 = pybuda.op.Relu("relu13", conv14)
        conv15 = self.conv15(relu13)
        relu14 = pybuda.op.Relu("relu14", conv15)
        conv16 = self.conv16(relu14)
        add12 = pybuda.op.Add("add12", conv16, mul12)

        # ... tail ...
        #   ... tail flow 1 ...
        W, Z, R, C = 1, 1, add11.shape[-3], add11.shape[-1] * add11.shape[-2]
        resh1 = pybuda.op.Reshape("resh1", add11, (W, Z, R, C))
        tr1 = pybuda.op.Transpose("tr1", resh1, -1, -2)
        red1 = pybuda.op.ReduceAvg("red1", tr1, -2)
        lin1 = self.lin1(red1)
        sm1 = pybuda.op.Softmax("sm1", lin1, dim=-1, stable=True)

        #   ... tail flow 2 ...
        W, Z, R, C = 1, 1, add12.shape[-3], add12.shape[-1] * add12.shape[-2]
        resh2 = pybuda.op.Reshape("resh2", add12, (W, Z, R, C))
        tr2 = pybuda.op.Transpose("tr2", resh2, -1, -2)
        red2 = pybuda.op.ReduceAvg("red2", tr2, -2)
        lin2 = self.lin2(red2)
        sm2 = pybuda.op.Softmax("sm2", lin2, dim=-1, stable=True)

        #   ... tail flow 3 ...
        W, Z, R, C = 1, 1, mul13.shape[-3], mul13.shape[-1] * mul13.shape[-2]
        resh3 = pybuda.op.Reshape("resh3", mul13, (W, Z, R, C))
        tr3 = pybuda.op.Transpose("tr3", resh3, -1, -2)
        red3 = pybuda.op.ReduceAvg("red3", tr3, -2)
        lin3 = self.lin3(red3)
        sm3 = pybuda.op.Softmax("sm3", lin3, dim=-1, stable=True)

        return sm1, sm2, sm3


TEST_F_DISABLE_PADDING_PASS_FLAG = True
TEST_F_PRINT_GRAPH_VIZ_FLAG = False
TEST_F_PRINT_GRAPH_AT_FLAG = False
TEST_F_FRACTURING_FLAG = False

TEST_F_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = True
TEST_F_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_F_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_F_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_F_SHAPE_SIZE_1_FLAG = False
TEST_F_SHAPE_SIZE_2_FLAG = False
TEST_F_SHAPE_SIZE_3_FLAG = False
TEST_F_SHAPE_SIZE_4_FLAG = False

TEST_F_KERNEL_SQUARED_ODD = True
TEST_F_KERNEL_SQUARED_EVEN = False

TEST_F_LOGGER_LEVEL_TRACE = False
TEST_F_LOGGER_LEVEL_DEBUG = False

TEST_F_INPUT_NO = 3


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_F_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    # Environment variable that allows printing a graph
    if TEST_F_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_F_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_F_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_F_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_F_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_F_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_F_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG:
        os.environ["PYBUDA_GRAPH_SOLVER_SELF_CUT_TYPE"] = "FastCut"
    if TEST_F_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
        os.environ["PYBUDA_LEGALIZER_DEBUG_NODE_NAME"] = ""



# The main reason why we use shapes of different sizes is
# because convolutional neural networks can't always work with big shapes.

# Each size category is bigger than previous one 
# for one or half order of magnitude

# Shapes -> Size 1
shapes_size_1 = [
    (12, 12),
    (16, 16),
    (17, 17),
    (73, 73),
    (64, 64),
    (32, 32),
    (37, 37)
]

# Shapes -> Size 2
shapes_size_2 = [
    (128, 128), 
    (192, 192), 
    (211, 211), 
    (212, 212),
    (224, 224),
    (337, 337),
    (359, 359)
]

# Shapes -> Size 3
shapes_size_3 = [
    (503, 503),
    (541, 541),
    (977, 977),
    (768, 768),
    (1024, 1024)
]

# Shapes -> Size 4
shapes_size_4 = [
    (2048, 2048),
    (3000, 3000),
    (4096, 4096)
]

original_shape = []

if TEST_F_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_F_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_F_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3
if TEST_F_SHAPE_SIZE_4_FLAG:
    original_shape += shapes_size_4

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
if TEST_F_KERNEL_SQUARED_ODD:
    kernels += kernel_squared_odd
if TEST_F_KERNEL_SQUARED_EVEN:
    kernels += kernel_squared_even

kernel_ids = [f"kernel={'x'.join([str(jtem) for jtem in item])}" for item in kernels]


@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.parametrize("in_features", [32])
@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("kernel", kernels, ids=kernel_ids)
def test_padding_pass_f(
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

    model = TestPaddingPassF(
                name=f"{TestPaddingPassF}_{original_shape[0]}_{original_shape[1]}",
                shape=original_shape,
                in_channels=in_channels,
                out_channels=out_channels,
                in_features=in_features,
                out_features=out_features,
                kernel=kernel
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

    inputs = [Tensor.create_from_torch(torch.rand(act_shape)) for _ in range(TEST_F_INPUT_NO)]

    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *inputs, 
        compiler_cfg=compiler_cfg, 
        verify_cfg=verify_cfg
    )