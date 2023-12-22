import pytest

from cv_demos.wideresnet.pytorch_wideresnet_timm import run_wideresnet_timm_pytorch
from cv_demos.wideresnet.pytorch_wideresnet_torchhub import run_wideresnet_torchhub_pytorch

variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.wideresnet
def test_wideresnet_torchhub_pytorch(clear_pybuda, variant):
    run_wideresnet_torchhub_pytorch(variant)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.wideresnet
def test_wideresnet_timm_pytorch(clear_pybuda, variant):
    run_wideresnet_timm_pytorch(variant)
