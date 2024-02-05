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
from pybuda._C.backend_api import BackendType, BackendDevice
from pybuda._C import DataFormat


class TestPaddingPassE(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on element-wise operation.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param7 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(3)]
        for i in range(1, 8):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):
        
        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x3, self.train_param3)
        add1 = pybuda.op.Add("add1", x1, x2)
        add2 = pybuda.op.Add("add2", x2, x3)

        # Layer 3
        mul4 = pybuda.op.Multiply("mul4", self.train_param1, add1)
        mul5 = pybuda.op.Multiply("mul5", self.train_param2, add2)
        mul6 = pybuda.op.Multiply("mul6", mul3, self.train_param3)
        mul7 = pybuda.op.Multiply("mul7", add2, self.train_param6)
        add3 = pybuda.op.Add("add3", mul1, mul4)
        add4 = pybuda.op.Add("add4", mul2, mul5)

        # Layer 4
        relu1 = pybuda.op.Relu("relu1", add3)
        relu2 = pybuda.op.Relu("relu2", mul4)
        relu3 = pybuda.op.Relu("relu3", add4)
        exp1 = pybuda.op.Exp("exp1", mul6)

        # Layer 5
        add5 = pybuda.op.Add("add5", mul5, mul6)
        add6 = pybuda.op.Add("add6", mul2, relu3)
        mul8 = pybuda.op.Multiply("mul8", self.train_param4, relu1)
        mul9 = pybuda.op.Multiply("mul9", self.train_param5, exp1)
        mul10 = pybuda.op.Multiply("mul10", relu1, mul2)
        mul11 = pybuda.op.Multiply("mul11", exp1, add4)
        mul12 = pybuda.op.Multiply("mul12", relu2, mul3)
        mul13 = pybuda.op.Multiply("mul13", add4, self.train_param7)
        mul14 = pybuda.op.Multiply("mul14", add6, add2)

        # Layer 6
        recip1 = pybuda.op.Reciprocal("recip1", mul10)
        exp2 = pybuda.op.Exp("exp2", mul11)
        exp3 = pybuda.op.Exp("exp3", add5)
        relu4 = pybuda.op.Relu("relu4", mul12)
        relu5 = pybuda.op.Relu("relu5", mul14)

        # Layer 7
        mul15 = pybuda.op.Multiply("mul15", add1, relu4)
        add7 = pybuda.op.Add("add7", mul9, mul8)
        add8 = pybuda.op.Add("add8", recip1, exp2)
        add9 = pybuda.op.Add("add9", exp1, exp3)
        add10 = pybuda.op.Add("add10", mul7, mul13)
        
        # Layer 8
        mul16 = pybuda.op.Multiply("mul16", exp1, add8)
        mul17 = pybuda.op.Multiply("mul17", add7, add10)
        mul18 = pybuda.op.Multiply("mul18", add9, mul15)
        mul19 = pybuda.op.Multiply("mul19", relu2, relu5)

        return mul16, mul17, mul18, mul19


class TestPaddingPassE_1(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on element-wise operation.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]
        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):
        
        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param1)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        mul4 = pybuda.op.Multiply("mul4", mul1, mul2)
        mul5 = pybuda.op.Multiply("mul5", mul2, mul3)

        # Layer 4
        mul6 = pybuda.op.Multiply("mul6", mul4, mul5)

        return mul6


class TestPaddingPassE_2(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on element-wise operation.

    def __init__(self, name: str, shape: list, explicit_padding: bool = False):
        super().__init__(name)
        self.shape = shape
        self.explicit_padding = explicit_padding
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # This explicit padding is for testing padding/unpadding pass with t-streaming
        explicit_padding = self.explicit_padding and self.shape[-2:] == (192, 192)
        
        # Layer 2
        if explicit_padding:
            pad1 = pybuda.op.BudaPad("pad1", x, (4, 4), 0.0)
            pad2 = pybuda.op.BudaPad("pad2", self.train_param, (4, 4), 0.0)
            mul = pybuda.op.Multiply("mul", pad1, pad2)
        else:
            mul = pybuda.op.Multiply("mul", x, self.train_param)

        # Layer 3
        exp = pybuda.op.Exp("exp", mul)

        if explicit_padding:
            unpad = pybuda.op.BudaUnpad("unpad", exp, (192, 192), (4, 4))
            return unpad
        else:
            return exp


class TestPaddingPassE_3(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on element-wise operation.

    def __init__(self, name: str, shape: list, explicit_padding: bool = False):
        super().__init__(name)
        self.shape = shape
        self.explicit_padding = explicit_padding
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # This explicit padding is for testing padding/unpadding pass with t-streaming
        explicit_padding = self.explicit_padding and (self.shape[-2:] == (192, 192) or self.shape[-2:] == (2000, 2000))
        
        # Layer 2
        if explicit_padding:
            if self.shape[-2:] == (192, 192):
                pad_rt, pad_ct = 4, 4
            elif self.shape[-2:] == (2000, 2000):
                pad_rt, pad_ct = 7, 7
            else:
                pad_rt, pad_ct = 0, 0
            pad1 = pybuda.op.BudaPad("pad1", x, (pad_rt, pad_ct), 0.0)
            pad2 = pybuda.op.BudaPad("pad2", self.train_param, (pad_rt, pad_ct), 0.0)
            mul1 = pybuda.op.Multiply("mul1", pad1, pad2)
        else:
            mul1 = pybuda.op.Multiply("mul1", x, self.train_param)

        # Layer 3
        if explicit_padding:
            mul2 = pybuda.op.Multiply("mul2", pad1, mul1)
            mul3 = pybuda.op.Multiply("mul3", mul1, pad2)
        else:
            mul2 = pybuda.op.Multiply("mul2", x, mul1)
            mul3 = pybuda.op.Multiply("mul3", mul1, self.train_param)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", mul1, mul3)

        # Layer 5
        mul5 = pybuda.op.Multiply("mul5", mul2, mul4)

        if explicit_padding:
            if self.shape[-2:] == (192, 192):
                orig_r, orig_c = 192, 192
                pad_rt, pad_ct = 4, 4
            elif self.shape[-2:] == (2000, 2000):
                orig_r, orig_c = 2000, 2000
                pad_rt, pad_ct = 7, 7
            else:
                pad_rt, pad_ct = 0, 0
            unpad = pybuda.op.BudaUnpad("unpad", mul5, (orig_r, orig_c), (pad_rt, pad_ct))
            return unpad
        else:
            return mul5


TEST_E_DISABLE_PADDING_PASS_FLAG = False
TEST_E_PADDING_PASS_ELEMENT_WISE_FLAG = True
TEST_E_PADDING_PASS_MATMUL_FLAG = True
TEST_E_PADDING_PASS_BUFFER_QUEUE_FLAG = False
TEST_E_PADDING_PASS_DISABLE_BUDA_OP_FLAG = False

TEST_E_PYBUDA_DISABLE_OP_FUSING = True

TEST_E_VERIFY_ALL_FLAG = True
TEST_E_PRINT_GRAPH_VIZ_FLAG = False
TEST_E_PRINT_GRAPH_AT_FLAG = False
TEST_E_FRACTURING_FLAG = False

TEST_E_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = False
TEST_E_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = False
TEST_E_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_E_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_E_SHAPE_SIZE_1_FLAG = False
TEST_E_SHAPE_SIZE_2_FLAG = True
TEST_E_SHAPE_SIZE_3_FLAG = True
TEST_E_SHAPE_SIZE_4_FLAG = False

TEST_E_MODEL_E_FLAG = False
TEST_E_MODEL_E_1_FLAG = False
TEST_E_MODEL_E_2_FLAG = True
TEST_E_MODEL_E_3_FLAG = True

TEST_E_LOGGER_LEVEL_TRACE = True
TEST_E_LOGGER_LEVEL_DEBUG = True


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_E_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    if TEST_E_PADDING_PASS_ELEMENT_WISE_FLAG:
        os.environ["PYBUDA_PADDING_PASS_ELEMENT_WISE"] = "1"
    if TEST_E_PADDING_PASS_MATMUL_FLAG:
        os.environ["PYBUDA_PADDING_PASS_MATMUL"] = "1"
    if TEST_E_PADDING_PASS_BUFFER_QUEUE_FLAG:
        os.environ["PYBUDA_PADDING_PASS_BUFFER_QUEUE"] = "1"
    if TEST_E_PADDING_PASS_DISABLE_BUDA_OP_FLAG:
        os.environ["PYBUDA_PADDING_PASS_DISABLE_BUDA_OP"] = "1"
    else:
        os.environ["PYBUDA_PADDING_PASS_DISABLE_BUDA_OP"] = "0"

    # Environment variables that controls operation fusing
    if TEST_E_PYBUDA_DISABLE_OP_FUSING:
        os.environ["PYBUDA_DISABLE_OP_FUSING"] = "1"

    # Environment variable that allows printing a graph
    if TEST_E_VERIFY_ALL_FLAG:
        os.environ["PYBUDA_FORCE_VERIFY_ALL"] = "1"
    if TEST_E_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_E_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_E_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_E_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_E_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_E_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_E_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
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
    # (1, 6, 32, 192), 
    # (1, 1, 48, 256), 
    # (8, 192, 192),
    # (128, 128), 
    # (192, 192), 
    # (211, 211), 
    # (212, 212),

    # ------------------------------------------------------------------- #
    # -------- TEST BACKEND PADDING CONSTRAINTS AND T-STREAMING --------- #
    # ------------------------------------------------------------------- #
    # R = 224 = 7 x 32 = 7 tiles, C = 224 = 7 x 32 = 7 tiles
    (224, 224),
    # ------------------------------------------------------------------- #

    # (337, 337),
    # (359, 359)
]

# Shapes -> Size 3
shapes_size_3 = [
    # (503, 503),
    # (541, 541),

    # ------------------------------------------------------------------- #
    # -------- TEST BACKEND PADDING CONSTRAINTS AND T-STREAMING --------- #
    # ------------------------------------------------------------------- #
    # R = 832 = 26 x 32 = 26 tiles, C = 832 = 26 x 32 = 26 tiles
    (832, 832),
    # R = 1472 = 46 x 32 = 46 tiles, C = 1472 = 46 x 32 = 46 tiles
    (1472, 1472),

    # R = 544 = 17 x 32 = 17 tiles, C = 544 = 17 x 32 = 17 tiles
    (544, 544),
    # R = 864 = 27 x 32 = 27 tiles, C = 864 = 27 x 32 = 27 tiles
    (864, 864),
    # R = 1184 = 37 x 32 = 37 tiles, C = 1184 = 37 x 32 = 37 tiles
    (1184, 1184),
    # R = 1504 = 47 x 32 = 47 tiles, C = 1504 = 47 x 32 = 47 tiles
    (1504, 1504),
    # R = 1824 = 57 x 32 = 57 tiles, C = 1824 = 57 x 32 = 57 tiles
    (1824, 1824),
    
    # R = 1216 = 38 x 32 = 38 tiles, C = 1216 = 38 x 32 = 38 tiles
    (1216, 1216),
    # R = 1856 = 58 x 32 = 58 tiles, C = 1856 = 58 x 32 = 58 tiles
    (1856, 1856),
    # ------------------------------------------------------------------- #

#     (977, 977),
#     (768, 768),
#     (1024, 1024)
]

# Shapes -> Size 4
shapes_size_4 = [

    # ------------------------------------------------------------------- #
    # -------- TEST BACKEND PADDING CONSTRAINTS AND T-STREAMING --------- #
    # ------------------------------------------------------------------- #
    # R = 2112 = 66 x 32 = 66 tiles, C = 2112 = 66 x 32 = 66 tiles
    (2112, 2112),
    # R = 2144 = 67 x 32 = 67 tiles, C = 2144 = 67 x 32 = 67 tiles
    (2144, 2144),
    # R = 2176 = 68 x 32 = 68 tiles, C = 2176 = 68 x 32 = 68 tiles
    (2176, 2176),
    # ------------------------------------------------------------------- #

    # (2000, 2000),
    # (2048, 2048),
    # (3000, 3000),
    # (4096, 4096)
]

original_shape = []

if TEST_E_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_E_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_E_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3
if TEST_E_SHAPE_SIZE_4_FLAG:
    original_shape += shapes_size_4

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]

test_model = []

if TEST_E_MODEL_E_FLAG:
    test_model.append(("TestPaddingPassE", TestPaddingPassE))
if TEST_E_MODEL_E_1_FLAG:
    test_model.append(("TestPaddingPassE_1", TestPaddingPassE_1))
if TEST_E_MODEL_E_2_FLAG:
    test_model.append(("TestPaddingPassE_2", TestPaddingPassE_2))
if TEST_E_MODEL_E_3_FLAG:
    test_model.append(("TestPaddingPassE_3", TestPaddingPassE_3))


# @pytest.mark.xfail
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("test_model", test_model)
def test_padding_pass_e(
    test_kind,
    test_device,
    original_shape,
    test_model
):
    if test_device.arch == BackendDevice.Wormhole_B0:
        pytest.skip("Skip until #731 is solved")

    if test_kind.is_training():
        pytest.skip()

    if test_kind.is_training():
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    verify_cfg = VerifyConfig(
        run_golden=False,
        verify_all=True
    )

    test_name = f"{test_model[0]}_{original_shape[0]}_{original_shape[1]}_{pass_name}"
    model = test_model[1](name=test_name, shape=original_shape)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
            enable_training=test_kind.is_training(),
            enable_t_streaming=False
        ), 
        verify_cfg=verify_cfg
    )


@pytest.mark.parametrize("explicit_padding", (True, False), ids=[True, False])
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
def test_padding_pass_e_argument_e2(
    test_kind,
    test_device,
    original_shape,
    explicit_padding
):

    if test_kind.is_training():
        pytest.skip()

    if explicit_padding:
        pytest.skip()

    if explicit_padding and test_kind.is_training():
        pytest.skip("Explicit padding is not supported in training mode, because backward pass is not implemented yet")

    if test_kind.is_training():
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    paddings = {
        'mul': True
    }

    verify_cfg = VerifyConfig(
        run_golden=False,
        verify_all=True
    )

    test_name = f"TestPaddingPassE_2_{'shape=' + 'x'.join([str(item) for item in original_shape])}_{pass_name}_explicit_padding={explicit_padding}"
    model = TestPaddingPassE_2(name=test_name, shape=original_shape, explicit_padding=explicit_padding)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
            enable_training=test_kind.is_training(),
            enable_t_streaming=False,
            paddings=paddings
        ), 
        verify_cfg=verify_cfg
    )


@pytest.mark.parametrize("explicit_padding", (True, False), ids=[True, False])
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
def test_padding_pass_e_argument_e3(
    test_kind,
    test_device,
    original_shape,
    explicit_padding
):
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        pytest.skip("Skip until #731 is solved")

    if explicit_padding:
        pytest.skip()

    if explicit_padding and test_kind.is_training():
        pytest.skip("Explicit padding is not supported in training mode, because backward pass is not implemented yet")

    if test_kind.is_training():
        pytest.skip()

    if test_kind.is_training():
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    paddings = {
        'mul1': True,
        'mul4': True
    }

    verify_cfg = VerifyConfig(
        run_golden=False,
        verify_all=True
    )

    test_name = f"TestPaddingPassE_3_{'shape=' + 'x'.join([str(item) for item in original_shape])}_{pass_name}_explicit_padding={explicit_padding}"
    model = TestPaddingPassE_3(name=test_name, shape=original_shape, explicit_padding=explicit_padding)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0,
        model.name, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
            enable_training=test_kind.is_training(),
            enable_t_streaming=False,
            paddings=paddings
        ), 
        verify_cfg=verify_cfg
    )
