# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic tests for padding
#
import pytest

import torch
import yaml
import random
import time
import os

import pybuda
import pybuda.op
from pybuda import (
    PyBudaModule,
    TTDevice,
    BackendDevice,
    BackendType,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
    PyTorchModule,
)

from pybuda.utils import align_up_tile
from pybuda.pybudaglobal import TILE_DIM

from pybuda.verify import TestKind, verify_module



class BudaPadTest1(PyBudaModule):
    """
    Test wrapper for padding pad
    """

    def __init__(self, name, paddings, value):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.value = value

    def forward(self, act):
        return pybuda.op.BudaPad("buda_pad", act, self.paddings, self.value)

class BudaPadTest2(PyBudaModule):
    """
    Test wrapper for padding pad
    """

    def __init__(self, name, paddings, value):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.value = value

    def forward(self, act1, act2):
        pad1 = pybuda.op.BudaPad("buda_pad1", act1, self.paddings, self.value)
        pad2 = pybuda.op.BudaPad("buda_pad2", act2, self.paddings, self.value)
        multiply = pybuda.op.Multiply("multiply", pad1, pad2)
        return multiply

@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("shape", ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200))])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in(0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 10), (11, 0), (11, 10)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
def test_buda_pad1(
    training,
    test_device,
    paddings,
    value,
    shape
):

    if training:
        pytest.skip() # no backward pass for padding pad

    test_name = f"buda_pad1_{str(paddings[0])}_{str(paddings[1])}_{value}_shape={'x'.join([str(item) for item in shape])}"

    mod = BudaPadTest1(name="buda_pad1", paddings=paddings, value=value)

    input_shapes = (shape, )

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            verify_all=True,
            run_golden=True
        )
    )

@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("shape", ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200))])
@pytest.mark.parametrize("value", (0.0, -0.5), ids=[f"value={str(value)}" for value in (0.0, -0.5)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 10), (11, 0), (11, 10)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
def test_buda_pad2(
    training,
    test_device,
    paddings,
    value,
    shape
):

    if training:
        pytest.skip() # no backward pass for padding pad

    test_name = f"buda_pad2_{str(paddings[0])}_{str(paddings[1])}_{value}_shape={'x'.join([str(item) for item in shape])}"

    mod = BudaPadTest2(name=test_name, paddings=paddings, value=value)

    input_shapes = (shape, shape)

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            verify_all=True,
            run_golden=True
        )
    )

class BudaUnpadTest1(PyBudaModule):
    """
    Test wrapper for padding unpad
    """

    def __init__(self, name, original_length, paddings):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.original_length = original_length

    def forward(self, act):
        return pybuda.op.BudaUnpad("buda_unpad", act, self.original_length, self.paddings)

class BudaUnpadTest2(PyBudaModule):
    """
    Test wrapper for padding unpad
    """

    def __init__(self, name, original_length, paddings):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.original_length = original_length

    def forward(self, act1, act2):
        unpad1 = pybuda.op.BudaUnpad("buda_unpad1", act1, self.original_length, self.paddings)
        unpad2 = pybuda.op.BudaUnpad("buda_unpad2", act2, self.original_length, self.paddings)
        multiply = pybuda.op.Multiply("multiply", unpad1, unpad2)
        return multiply

@pytest.mark.parametrize("original_shape", ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200))])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 10), (11, 0), (11, 10)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
def test_buda_unpad1(
    training,
    test_device,
    paddings,
    original_shape
):

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (original_shape[-2], original_shape[-1])

    shape = list(original_shape)
    shape[-1] = (align_up_tile(original_shape[-1]) // TILE_DIM + paddings[-1]) * TILE_DIM
    shape[-2] = (align_up_tile(original_shape[-2]) // TILE_DIM + paddings[-2]) * TILE_DIM
    input_shapes = (shape, )

    test_name = f"buda_unpad1_{str(paddings[0])}_{str(paddings[1])}_orig_shape={'x'.join([str(item) for item in original_shape])}_shape={'x'.join([str(item) for item in shape])}"

    mod = BudaUnpadTest1(name=test_name, original_length=original_length, paddings=paddings)

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            verify_all=True,
            run_golden=True
        )
    )

@pytest.mark.parametrize("original_shape", ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 200, 300), (3, 200, 300), (1, 5, 100, 200))])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 10), (11, 0), (11, 10)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
def test_buda_unpad2(
    training,
    test_device,
    paddings,
    original_shape
):

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (original_shape[-2], original_shape[-1])

    shape = list(original_shape)
    shape[-1] = (align_up_tile(original_shape[-1]) // TILE_DIM + paddings[-1]) * TILE_DIM
    shape[-2] = (align_up_tile(original_shape[-2]) // TILE_DIM + paddings[-2]) * TILE_DIM
    input_shapes = (shape, shape)

    test_name = f"buda_unpad2_{str(paddings[0])}_{str(paddings[1])}_orig_shape={'x'.join([str(item) for item in original_shape])}_shape={'x'.join([str(item) for item in shape])}"

    mod = BudaUnpadTest2(name=test_name, original_length=original_length, paddings=paddings)

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            verify_all=True,
            run_golden=True
        )
    )

class PaddingTest1(PyBudaModule):
    """
    Test wrapper for padding pad and unpad
    This test contains a both padding pad and unpad
    """

    def __init__(self, name, paddings, value, original_length):
        super().__init__(name)
        self.paddings = paddings
        self.value = value
        self.original_length = original_length

    def forward(self, x1, x2):

        # pad inputs, x1 and x2
        pad_x1 = pybuda.op.BudaPad("buda_pad1", x1, self.paddings, self.value)
        pad_x2 = pybuda.op.BudaPad("buda_pad2", x2, self.paddings, self.value)

        # multiply padded inputs
        multiply = pybuda.op.Multiply("multiply", pad_x1, pad_x2)

        # unpad the result of the multiplication
        unpad_multiply = pybuda.op.BudaUnpad("buda_unpad", multiply, self.original_length, self.paddings)

        return unpad_multiply

class PaddingTest2(PyBudaModule):
    """
    Test wrapper for padding pad and unpad
    This test contains a both padding pad and unpad and matmul operation
    """

    def __init__(self, name, paddings, value, original_length):
        super().__init__(name)
        self.paddings = paddings
        self.value = value
        self.original_length = original_length

    def forward(self, x1, x2):

        # pad inputs, x1 and x2
        pad1 = pybuda.op.BudaPad("buda_pad1", x1, self.paddings, self.value)
        pad2 = pybuda.op.BudaPad("buda_pad2", x2, self.paddings, self.value)

        # add padded inputs
        add = pybuda.op.Add("add", pad1, pad2)

        # unpad the result of the addition
        unpad_add = pybuda.op.BudaUnpad("unpad_add", add, self.original_length, self.paddings)

        # pad again the result of the addition
        pad3 = pybuda.op.BudaPad("buda_pad3", unpad_add, self.paddings, self.value)

        exp = pybuda.op.Exp("exp", x1)

        # matrix multiplication between padded and unpadded inputs
        mm = pybuda.op.Matmul("matmul", pad3, exp)

        # unpad the result of the matrix multiplication
        unpad_mm = pybuda.op.BudaUnpad("unpad_mm", mm, self.original_length, self.paddings)

        return unpad_mm

@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("shape", ((1, 200, 300), (3, 200, 300), (1, 5, 400, 200)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 200, 300), (3, 200, 300), (1, 5, 400, 200))])
@pytest.mark.parametrize("value", (0.0, -0.5), ids=[f"value={str(value)}" for value in (0.0, -0.5)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
def test_padding1(
    training,
    test_device,
    paddings,
    value,
    shape
):

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (shape[-2], shape[-1])
    test_name = f"padding1_{str(paddings[0])}_{str(paddings[1])}_shape={'x'.join([str(item) for item in shape])}_value={str(value)}"
    input_shapes = (shape, shape)

    mod = PaddingTest1(
        name=test_name, 
        original_length=original_length, 
        paddings=paddings, 
        value=value
    )

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            verify_all=True,
            run_golden=True
        )
    )


@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("shape", ((1, 800, 800), (3, 700, 700), (1, 5, 200, 200)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 800, 800), (3, 700, 700), (1, 5, 200, 200))])
@pytest.mark.parametrize("value", (0.0, -0.5), ids=[f"value={str(value)}" for value in (0.0, -0.5)])
@pytest.mark.parametrize("paddings", ((0, 0), (5, 0)), ids=["no-padding", "R-padding"])
def test_padding2(
    training,
    test_device,
    paddings,
    value,
    shape
):
    if test_device.arch == BackendDevice.Wormhole_B0 and shape == (3, 700, 700):
        pytest.skip("Skip until #731 is solved")

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (shape[-2], shape[-1])
    test_name = f"padding2_{str(paddings[0])}_{str(paddings[1])}_shape={'x'.join([str(item) for item in shape])}_value={str(value)}"
    input_shapes = (shape, shape)

    mod = PaddingTest2(
        name=test_name, 
        original_length=original_length, 
        paddings=paddings, 
        value=value
    )

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            verify_all=True,
            run_golden=True
        )
    )

