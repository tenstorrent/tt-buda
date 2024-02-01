# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import torch

import pybuda
import pybuda.op
from pybuda import (
    Tensor,
    VerifyConfig,
    DataFormat,
    MathFidelity,
)
from pybuda.verify import TestKind
from test.common import run_torch

# Skip backend compilation pending bbe#1442
os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"
data_formats = {
    DataFormat.Int8 : "Int8",
    DataFormat.Float32: "Float32",
    DataFormat.Float16: "Float16",
    DataFormat.Float16_b: "Float16_b",
    DataFormat.Bfp8: "Bfp8",
    DataFormat.Bfp8_b: "Bfp8_b",
}


@pytest.mark.parametrize(
    "math_fidelity",
    (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi3),
    ids=["LoFi", "HiFi2", "HiFi3", "HiFi4"],
)
def test_int8_math_fidelity(test_device, math_fidelity):
    # math-fidelity must be set to HiFi4
    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        input_df={
            0: [pybuda.DataFormat.Int8, True],
            1: [pybuda.DataFormat.Int8, True],
        },
        math_fidelity=math_fidelity,
        accumulate_df=pybuda.DataFormat.Int8,
        output_df=pybuda.DataFormat.Int8,
    )

    @run_torch(
        VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )
    def matmul_int8_math_fidelity(x, y):
        return torch.matmul(x, y)

    x = Tensor.create_from_torch(
        torch.randn(1, 1, 32, 32), dev_data_format=DataFormat.Int8
    )
    y = Tensor.create_from_torch(
        torch.randn(1, 1, 32, 32), dev_data_format=DataFormat.Int8
    )
    matmul_int8_math_fidelity(x, y)


@pytest.mark.parametrize("accumulate_df", data_formats.keys(), ids=data_formats.values())
@pytest.mark.parametrize("intermediate_df", data_formats.keys(), ids=data_formats.values())
@pytest.mark.parametrize("output_df", data_formats.keys(), ids=data_formats.values())
def test_int8_dfs(test_device, accumulate_df, intermediate_df, output_df):
    # acc_df must be set to int8
    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        input_df={
            0: [pybuda.DataFormat.Int8, True],
            1: [pybuda.DataFormat.Int8, True],
        },
        math_fidelity=MathFidelity.HiFi4,
        accumulate_df=accumulate_df,
        intermediate_df=intermediate_df,
        output_df=output_df,
    )

    @run_torch(
        VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
    )
    def matmul_int8_dfs(x, y):
        return torch.matmul(x, y)

    x = Tensor.create_from_torch(
        torch.randn(1, 1, 32, 32), dev_data_format=DataFormat.Int8
    )
    y = Tensor.create_from_torch(
        torch.randn(1, 1, 32, 32), dev_data_format=DataFormat.Int8
    )
    matmul_int8_dfs(x, y)
