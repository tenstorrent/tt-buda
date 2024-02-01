# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from pybuda.verify.backend import verify_module
from pybuda import VerifyConfig
from pybuda.verify.config import TestKind, NebulaGalaxy

from test.model_demos.models.deit import generate_model_deit_imgcls_hf_pytorch

variants = ["facebook/deit-base-patch16-224", "facebook/deit-base-distilled-patch16-224", "facebook/deit-small-patch16-224", "facebook/deit-tiny-patch16-224"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_base_classify_224_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_deit_imgcls_hf_pytorch(
        test_device, variant,
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
            chip_ids=NebulaGalaxy.chip_ids if "PYBUDA_NEB_GALAXY_CI" in os.environ and int(os.environ.get("PYBUDA_NEB_GALAXY_CI"))==1 else [0],
            pcc=0.78
        )
    )
