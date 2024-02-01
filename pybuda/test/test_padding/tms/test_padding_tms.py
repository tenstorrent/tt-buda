# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic tests for padding with TMs
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


class BudaPadTMsTest1(PyBudaModule):
    
    def __init__(self, name, paddings, value, vfactor, hfactor):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.value = value
        # Vertical and horizontal slicing factors
        self.vfactor = vfactor
        self.hfactor = hfactor
        # Transpose dimensions, transpose is always applied to the last two dimensions
        self.dim1 = -2
        self.dim2 = -1

    def forward(self, act):

        vsl = pybuda.op.VSlice("vsl", act, self.vfactor)
        hsl = pybuda.op.HSlice("hsl", vsl, self.hfactor)
        tr = pybuda.op.Transpose("tr", hsl, self.dim1, self.dim2)
        pad = pybuda.op.BudaPad("buda_pad", tr, self.paddings, self.value)

        return pad

class BudaPadTMsTest2(PyBudaModule):
    
    def __init__(self, name, paddings, value, vfactor, hfactor):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.value = value
        # Vertical and horizontal slicing factors
        self.vfactor = vfactor
        self.hfactor = hfactor
        # Transpose dimensions, transpose is always applied to the last two dimensions
        self.dim1 = -2
        self.dim2 = -1

    def forward(self, act1, act2):

        # Flow of the first input
        vsl1 = pybuda.op.VSlice("vsl1", act1, self.vfactor)
        hsl1 = pybuda.op.HSlice("hsl1", vsl1, self.hfactor)
        pad1 = pybuda.op.BudaPad("buda_pad1", hsl1, self.paddings, self.value)

        # Flow of the second input
        hsl2 = pybuda.op.HSlice("hsl2", act2, self.hfactor)
        vsl2 = pybuda.op.VSlice("vsl2", hsl2, self.vfactor)
        pad2 = pybuda.op.BudaPad("buda_pad2", vsl2, self.paddings, self.value)

        # Merge the two flows
        mul = pybuda.op.Multiply("mul", pad1, pad2)
        tr = pybuda.op.Transpose("tr", mul, self.dim1, self.dim2)
        vst = pybuda.op.VStack("vst", tr, self.vfactor)

        return vst

@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("hfactor", (2, 3), ids=[f"hfactor={str(hfactor)}" for hfactor in (2, 3)])
@pytest.mark.parametrize("vfactor", (2, 3), ids=[f"vfactor={str(vfactor)}" for vfactor in (2, 3)])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in (0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
@pytest.mark.parametrize("shape", ((1, 600, 600), (1, 3, 300, 300)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 600, 600), (1, 3, 300, 300))])
def test_buda_pad_tms1(
    training,
    test_device,
    shape,
    paddings,
    value,
    vfactor,
    hfactor
):

    if training:
        pytest.skip() # no backward pass for padding pad

    test_name  = f"buda_pad_tms1_{str(paddings[0])}_{str(paddings[1])}_{value}"
    test_name += f"_shape={'x'.join([str(item) for item in shape])}"
    test_name += f"_vfactor={str(vfactor)}_hfactor={str(hfactor)}"

    mod = BudaPadTMsTest1(
        name=test_name, 
        paddings=paddings, 
        value=value,
        vfactor=vfactor,
        hfactor=hfactor
    )

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
@pytest.mark.parametrize("hfactor", (2, 3), ids=[f"hfactor={str(hfactor)}" for hfactor in (2, 3)])
@pytest.mark.parametrize("vfactor", (2, 3), ids=[f"vfactor={str(vfactor)}" for vfactor in (2, 3)])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in (0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
@pytest.mark.parametrize("shape", ((1, 600, 600), (1, 3, 300, 300)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 600, 600), (1, 3, 300, 300))])
def test_buda_pad_tms2(
    training,
    test_device,
    shape,
    paddings,
    value,
    vfactor,
    hfactor
):

    if training:
        pytest.skip() # no backward pass for padding pad

    test_name  = f"buda_pad_tms2_{str(paddings[0])}_{str(paddings[1])}_{value}"
    test_name += f"_shape={'x'.join([str(item) for item in shape])}"
    test_name += f"_vfactor={str(vfactor)}_hfactor={str(hfactor)}"

    mod = BudaPadTMsTest2(
        name=test_name,
        paddings=paddings,
        value=value,
        vfactor=vfactor,
        hfactor=hfactor
    )

    input_shapes = (shape, shape, )

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

class BudaUnpadTMsTest1(PyBudaModule):
    
    def __init__(self, name, paddings, original_length, vfactor, hfactor):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.original_length = original_length
        # Vertical and horizontal slicing factors
        self.vfactor = vfactor
        self.hfactor = hfactor

    def forward(self, act):

        unpad = pybuda.op.BudaUnpad("buda_unpad", act, self.original_length, self.paddings)
        vsl = pybuda.op.VSlice("vsl", unpad, self.vfactor)
        hsl = pybuda.op.HSlice("hsl", vsl, self.hfactor)
        vst = pybuda.op.VStack("vst", hsl, self.vfactor)
        hst = pybuda.op.HStack("hst", vst, self.hfactor)

        return hst

class BudaUnpadTMsTest2(PyBudaModule):
    
    def __init__(self, name, paddings, original_length, vfactor, hfactor):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.original_length = original_length
        # Vertical and horizontal slicing factors
        self.vfactor = vfactor
        self.hfactor = hfactor
        # Transpose dimensions, transpose is always applied to the last two dimensions
        self.dim1 = -2
        self.dim2 = -1

    def forward(self, act1, act2):

        # Flow of the first input
        unpad1 = pybuda.op.BudaUnpad("buda_unpad1", act1, self.original_length, self.paddings)
        vsl1 = pybuda.op.VSlice("vsl1", unpad1, self.vfactor)
        tr1 = pybuda.op.Transpose("tr1", vsl1, self.dim1, self.dim2)

        # Flow of the second input
        unpad2 = pybuda.op.BudaUnpad("buda_unpad2", act2, self.original_length, self.paddings)
        vsl2 = pybuda.op.VSlice("vsl2", unpad2, self.vfactor)
        tr2 = pybuda.op.Transpose("tr2", vsl2, self.dim1, self.dim2)

        # Merge the two flows
        add = pybuda.op.Add("add", tr1, tr2)
        tr3 = pybuda.op.Transpose("tr3", add, self.dim1, self.dim2)
        vst = pybuda.op.VStack("vst", tr3)

        return vst

@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("hfactor", (2, 3), ids=[f"hfactor={str(hfactor)}" for hfactor in (2, 3)])
@pytest.mark.parametrize("vfactor", (2, 3), ids=[f"vfactor={str(vfactor)}" for vfactor in (2, 3)])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in (0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
@pytest.mark.parametrize("original_shape", ((1, 300, 300), (1, 3, 300, 300)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 300, 300), (1, 3, 300, 300))])
def test_buda_unpad_tms1(
    training,
    test_device,
    original_shape,
    paddings,
    value,
    vfactor,
    hfactor
):

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (original_shape[-2], original_shape[-1])

    shape = list(original_shape)
    shape[-1] = (align_up_tile(original_shape[-1]) // TILE_DIM + paddings[-1]) * TILE_DIM
    shape[-2] = (align_up_tile(original_shape[-2]) // TILE_DIM + paddings[-2]) * TILE_DIM
    input_shapes = (shape, )

    test_name  = f"buda_unpad_tms1_{str(paddings[0])}_{str(paddings[1])}_{value}"
    test_name += f"_shape={'x'.join([str(item) for item in shape])}"
    test_name += f"_vfactor={str(vfactor)}_hfactor={str(hfactor)}"

    mod = BudaUnpadTMsTest1(
        name=test_name, 
        paddings=paddings, 
        original_length=original_length,
        vfactor=vfactor,
        hfactor=hfactor
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
@pytest.mark.parametrize("hfactor", (2, 3), ids=[f"hfactor={str(hfactor)}" for hfactor in (2, 3)])
@pytest.mark.parametrize("vfactor", (2, 3), ids=[f"vfactor={str(vfactor)}" for vfactor in (2, 3)])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in (0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
@pytest.mark.parametrize("original_shape", ((1, 300, 300), (1, 3, 300, 300)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 300, 300), (1, 3, 300, 300))])
def test_buda_unpad_tms2(
    training,
    test_device,
    original_shape,
    paddings,
    value,
    vfactor,
    hfactor
):

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (original_shape[-2], original_shape[-1])

    shape = list(original_shape)
    shape[-1] = (align_up_tile(original_shape[-1]) // TILE_DIM + paddings[-1]) * TILE_DIM
    shape[-2] = (align_up_tile(original_shape[-2]) // TILE_DIM + paddings[-2]) * TILE_DIM
    input_shapes = (shape, shape, )

    test_name  = f"buda_unpad_tms2_{str(paddings[0])}_{str(paddings[1])}_{value}"
    test_name += f"_shape={'x'.join([str(item) for item in shape])}"
    test_name += f"_vfactor={str(vfactor)}_hfactor={str(hfactor)}"

    mod = BudaUnpadTMsTest2(
        name=test_name, 
        paddings=paddings, 
        original_length=original_length,
        vfactor=vfactor,
        hfactor=hfactor
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

class PaddingTMsTest1(PyBudaModule):

    def __init__(self, name, paddings, value, original_length, vfactor, hfactor):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.value = value
        self.original_length = original_length
        # Vertical and horizontal slicing factors
        self.vfactor = vfactor
        self.hfactor = hfactor
        # Transpose dimensions, transpose is always applied to the last two dimensions
        self.dim1 = -2
        self.dim2 = -1

    def forward(self, act1, act2):

        # Flow of the first input
            # Padding
        pad1 = pybuda.op.BudaPad("pad1", act1, self.paddings, self.value)
        exp = pybuda.op.Exp("exp", pad1)
        unpad1 = pybuda.op.BudaUnpad("unpad1", exp, self.original_length, self.paddings)
            # TMs after padding
        vsl1 = pybuda.op.VSlice("vsl1", unpad1, self.vfactor)
            # Padding before joining the two flows
        pad3 = pybuda.op.BudaPad("pad3", vsl1, self.paddings, self.value)

        # Flow of the second input
            # Padding
        pad2 = pybuda.op.BudaPad("pad2", act2, self.paddings, self.value)
        relu = pybuda.op.Relu("relu", pad2)
        unpad2 = pybuda.op.BudaUnpad("unpad2", relu, self.original_length, self.paddings)
            # TMs after padding
        vsl2 = pybuda.op.VSlice("vsl2", unpad2, self.vfactor)
            # Padding before joining the two flows
        pad4 = pybuda.op.BudaPad("pad4", vsl2, self.paddings, self.value)

        original_length_v = list(self.original_length)
        original_length_v[-2] //= self.vfactor

        # Merge the two flows
        add = pybuda.op.Add("add", pad3, pad4)
            # Unpad after joining the two flows
        unpad3 = pybuda.op.BudaUnpad("unpad3", add, original_length_v, self.paddings)
            # TMs after unpadding
        vst = pybuda.op.VStack("vst", unpad3, self.vfactor)

        return vst

class PaddingTMsTest2(PyBudaModule):

    def __init__(self, name, paddings, value, original_length, vfactor, hfactor):
        super().__init__(name)
        self.name = name
        self.paddings = paddings
        self.value = value
        self.original_length = original_length
        # Vertical and horizontal slicing factors
        self.vfactor = vfactor
        self.hfactor = hfactor
        # Transpose dimensions, transpose is always applied to the last two dimensions
        self.dim1 = -2
        self.dim2 = -1

    def forward(self, act1, act2):

        # Flow of the first input
            # TMs before padding
        vsl1 = pybuda.op.VSlice("vsl1", act1, self.vfactor)
        tr1 = pybuda.op.Transpose("tr1", vsl1, self.dim1, self.dim2)

        original_length_flow1 = list(self.original_length)
        original_length_flow1[-1] //= self.vfactor

            # Padding
        pad1 = pybuda.op.BudaPad("pad1", tr1, self.paddings, self.value)
        exp = pybuda.op.Exp("exp", pad1)
        unpad1 = pybuda.op.BudaUnpad("unpad1", exp, original_length_flow1, self.paddings)
            # TMs after padding
        tr2 = pybuda.op.Transpose("tr2", unpad1, self.dim1, self.dim2)
        vst1 = pybuda.op.VStack("vst1", tr2, self.vfactor)
        vsl3 = pybuda.op.VSlice("vsl3", vst1, self.vfactor)
        tr3 = pybuda.op.Transpose("tr3", vsl3, self.dim1, self.dim2)

        # Flow of the second input
            # TMs before padding
        hst1 = pybuda.op.HStack("hst1", act2, self.hfactor)
        vsl2 = pybuda.op.VSlice("vsl2", hst1, self.vfactor)

        original_length_flow2 = list(self.original_length)
        original_length_flow2[-2] //= self.vfactor
        original_length_flow2[-1] *= self.hfactor

            # Padding
        pad2 = pybuda.op.BudaPad("pad2", vsl2, self.paddings, self.value)
        relu = pybuda.op.Relu("relu", pad2)
        unpad2 = pybuda.op.BudaUnpad("unpad2", relu, original_length_flow2, self.paddings)
            # TMs after padding
        vst2 = pybuda.op.VStack("vst2", unpad2, self.vfactor)
        hsl1 = pybuda.op.HSlice("hsl1", vst2, self.hfactor)
        vsl4 = pybuda.op.VSlice("vsl4", hsl1, self.vfactor)
        tr4 = pybuda.op.Transpose("tr4", vsl4, self.dim1, self.dim2)

        # Merge the two flows

        original_length_flow_merged = list(self.original_length)
        original_length_flow_merged[-1] //= self.vfactor

            # Padding before joining the two flows
        pad3 = pybuda.op.BudaPad("pad3", tr3, self.paddings, self.value)
        pad4 = pybuda.op.BudaPad("pad4", tr4, self.paddings, self.value)
        add = pybuda.op.Add("add", pad3, pad4)
            # Unpad after joining the two flows
        unpad3 = pybuda.op.BudaUnpad("unpad3", add, original_length_flow_merged, self.paddings)
            # TMs after unpadding
        hst2 = pybuda.op.HStack("hst2", unpad3, self.vfactor)
        tr5 = pybuda.op.Transpose("tr5", hst2, self.dim1, self.dim2)

        return tr5

@pytest.mark.xfail(reason="Unsupported TM op pad! Found on op buda_pad, type nop, input 0. Backend should be updated.")
@pytest.mark.parametrize("hfactor", (2, 3), ids=[f"hfactor={str(hfactor)}" for hfactor in (2, 3)])
@pytest.mark.parametrize("vfactor", (2, 3), ids=[f"vfactor={str(vfactor)}" for vfactor in (2, 3)])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in (0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
@pytest.mark.parametrize("shape", ((1, 630, 630), (1, 3, 630, 630)), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 630, 630), (1, 3, 630, 630))])
def test_padding_tms1(
    training,
    test_device,
    shape,
    paddings,
    value,
    vfactor,
    hfactor
):

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (shape[-2], shape[-1])
    test_name  = f"padding_tms1_{str(paddings[0])}_{str(paddings[1])}_{value}"
    test_name += f"_shape={'x'.join([str(item) for item in shape])}"
    test_name += f"_vfactor={str(vfactor)}_hfactor={str(hfactor)}"
    input_shapes = (shape, shape, )

    mod = PaddingTMsTest1(
        name=test_name, 
        original_length=original_length, 
        paddings=paddings, 
        value=value,
        vfactor=vfactor,
        hfactor=hfactor
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

@pytest.mark.parametrize("hfactor", (2, ), ids=[f"hfactor={str(hfactor)}" for hfactor in (2, )])
@pytest.mark.parametrize("vfactor", (2, 3), ids=[f"vfactor={str(vfactor)}" for vfactor in (2, 3)])
@pytest.mark.parametrize("value", (0.0, 1.0), ids=[f"value={str(value)}" for value in (0.0, 1.0)])
@pytest.mark.parametrize("paddings", ((0, 0), (0, 5), (5, 0), (5, 5)), ids=["no-padding", "C-padding", "R-padding", "RC-padding"])
@pytest.mark.parametrize("shape", ((1, 6, 630, 630), ), ids=[f"shape={'x'.join([str(dim) for dim in shape])}" for shape in ((1, 6, 630, 630), )])
def test_padding_tms2(
    training,
    test_device,
    shape,
    paddings,
    value,
    vfactor,
    hfactor
):

    pytest.skip()   # WIP

    if training:
        pytest.skip() # no backward pass for padding pad

    original_length = (shape[-2], shape[-1])
    test_name  = f"padding_tms2_{str(paddings[0])}_{str(paddings[1])}_{value}"
    test_name += f"_shape={'x'.join([str(item) for item in shape])}"
    test_name += f"_vfactor={str(vfactor)}_hfactor={str(hfactor)}"
    input_shapes = (shape, shape, )

    mod = PaddingTMsTest2(
        name=test_name, 
        original_length=original_length, 
        paddings=paddings, 
        value=value,
        vfactor=vfactor,
        hfactor=hfactor
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
