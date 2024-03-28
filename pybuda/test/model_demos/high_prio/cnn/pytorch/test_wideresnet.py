# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from test.model_demos.models.wideresnet import (
    generate_model_wideresnet_imgcls_pytorch,
    generate_model_wideresnet_imgcls_timm,
)
from pybuda._C.backend_api import BackendDevice

variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_pytorch(variant, test_device):
    (
        model,
        inputs,
    ) = generate_model_wideresnet_imgcls_pytorch(
        test_device,
        variant,
    )

    os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            num_chips=1,
            pcc=0.97 if test_device.arch == BackendDevice.Grayskull else 0.99,
        ),
    )


variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_timm(variant, test_device):
    (
        model,
        inputs,
    ) = generate_model_wideresnet_imgcls_timm(
        test_device,
        variant,
    )

    os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"

    verify_module(
        model,
        input_shapes=[(inputs[0].shape,)],
        inputs=[(inputs[0],)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            num_chips=1,
            pcc=0.965 if test_device.arch == BackendDevice.Grayskull else 0.97,
        ),
    )
