# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of where operator
#

import pytest

import torch

import pybuda
import pybuda.op
import pybuda.tensor

from pybuda import PyBudaModule, VerifyConfig
from pybuda.verify import TestKind, verify_module

@pytest.mark.skip(reason="This test is failing due to not supporting 'BoolTensor' for a condition")
def test_cond_bool_tensor_manual_inputs(test_device):
    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, cond, x, y):
            return pybuda.op.Where("Where0", cond, x, y)

    mod = Model("where_test_model")

    # manual creation of input tensors
    # contidion_tensor is a boolean tensor what it should be
    condition_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[1, 0],
                       [1, 0],
                       [1, 0]]], dtype=torch.bool)
    )
    x_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[0.1490, 0.3861],
                       [1.4934, 0.4805],
                       [-0.3992, -1.1574]]])
    )
    y_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[1.0, 1.0],
                       [1.0, 1.0],
                       [1.0, 1.0]]])
    )

    verify_module(
        mod,
        input_shapes=None,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        inputs=[(condition_tensor, x_tensor, y_tensor)],
    )

@pytest.mark.skip(reason="This test is failing when condition_tensor elements have values <> 0.0 or 1.0")
def test_cond_non_bool_tensor_manual_inputs(test_device):
    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, cond, x, y):
            return pybuda.op.Where("Where0", cond, x, y)

    mod = Model("where_test_model")

    condition_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[0.2, 1.0],
                       [0.0, 1.0],
                       [1.1, 1.0]]])
    )
    x_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[0.1490, 0.3861],
                       [1.4934, 0.4805],
                       [-0.3992, -1.1574]]])
    )
    y_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[1.0, 1.0],
                       [1.0, 1.0],
                       [1.0, 1.0]]])
    )

    verify_module(
        mod,
        input_shapes=None,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        inputs=[(condition_tensor, x_tensor, y_tensor)],
    )

@pytest.mark.skip(reason="This test is failing due assertion error - data mismatch detected")
@pytest.mark.parametrize("input_shape", [(1, 3, 3)])
def test_where_input_shapes(test_device, input_shape):
    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, cond, x, y):
            return pybuda.op.Where("Where0", cond, x, y)

    mod = Model("where_test_model")
    input_shapes = tuple([input_shape for _ in range(3)])

    verify_module(
        mod,
        input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )

# Manually test where operator with PyTorch and PyBuda.
# Results are same for both, but verify_module fails due to different pcc values.

# working
cond_values_1 = [[[0., 0.],
                  [1., 0.],
                  [1., 0.]]]

# not working
cond_values_2 = [[[0.2, 0.],
                  [1., 0.],
                  [1., 0.]]]

@pytest.mark.skip(reason="This test is failing due to verify_module calculates wrong pcc")
@pytest.mark.parametrize("cond_values", [cond_values_1, cond_values_2])
def test_where_verify_module(test_device, cond_values):
    class Model(PyBudaModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, cond, x, y):
            v = pybuda.op.Where("Where0", cond, x, y)
            # PyBuda always works as expected:
            print(f"\n\nPyBuda output value: {v}\n\n")
            return v

    mod = Model("where_test_model")

    condition_torch = torch.tensor(cond_values, dtype=torch.bool)   # torch works only with bool type - explicit define dtype  
    condition_buda = pybuda.tensor.Tensor.create_from_torch(torch.tensor(cond_values))  # buda can work also with other types

    print(f"condition_torch:\n{condition_torch}")  # condition is a boolean tensor
    print(f"condition_buda:\n{condition_buda}")    # condition is a float tensor

    x_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[1000., 1000.],
                       [1000., 1000.],
                       [1000., 1000.]]])
    )
    y_tensor = pybuda.tensor.Tensor.create_from_torch(
        torch.tensor([[[5.0, 5.0],
                       [5.0, 5.0],
                       [5.0, 5.0]]])
    )    

    result_torch = torch.where(condition_torch, pybuda.tensor.Tensor.to_pytorch(x_tensor), pybuda.tensor.Tensor.to_pytorch(y_tensor))
    print(f"result_torch:\n{result_torch}")
    result_buda = pybuda.op.Where("Where0", condition_buda, x_tensor, y_tensor)
    print(f"result_buda:\n{result_buda}")

    output_are_the_same = torch.eq(result_torch, pybuda.tensor.Tensor.to_pytorch(result_buda)).all()
    print(f"\nAre results equal: {output_are_the_same}")
    if not output_are_the_same:
        # never failing here
        pytest.fail("Results are not equal")

    # verify_module calculates wrong pcc value for failing case
    # This is the error message for failing case:
    #    "AssertionError: Data mismatch on iteration 0 - Eval Output 0. PCC got 0.8087360843031886, required=0.99"
    #    ...
    #    "1 failed, 1 passed in 0.89s"
    verify_module(
        mod,
        input_shapes=None,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        inputs=[(condition_buda, x_tensor, y_tensor)],
    )
