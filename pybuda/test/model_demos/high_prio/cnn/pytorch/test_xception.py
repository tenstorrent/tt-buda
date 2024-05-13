# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind
from pybuda._C.backend_api import BackendDevice
from test.model_demos.models.xception import generate_model_xception_imgcls_timm

variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xception_timm(variant, test_device):
    if test_device.arch == BackendDevice.Grayskull and variant == "xception":
        os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"
    (
        model,
        inputs,
    ) = generate_model_xception_imgcls_timm(
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
        ),
    )
