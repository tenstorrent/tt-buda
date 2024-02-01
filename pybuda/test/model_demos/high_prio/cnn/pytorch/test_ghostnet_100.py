# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from test.model_demos.models.ghostnet import generate_model_ghostnet_imgcls_timm
from pybuda._C.backend_api import BackendDevice

variants = ["ghostnet_100"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_ghostnet_timm(variant, test_device):
    (
        model,
        inputs,
    ) = generate_model_ghostnet_imgcls_timm(
        test_device,
        variant,
    )

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
            pcc=0.965 if test_device.arch == BackendDevice.Grayskull else 0.99,
        ),
    )
