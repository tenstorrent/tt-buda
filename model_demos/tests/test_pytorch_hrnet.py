import pytest
from cv_demos.hrnet.pytorch_hrnet_osmr import run_hrnet_osmr_pytorch
from cv_demos.hrnet.pytorch_hrnet_timm import run_hrnet_timm_pytorch

variants_osmr = [
    "hrnet_w18_small_v1",
    "hrnet_w18_small_v2",
    "hrnetv2_w18",
    "hrnetv2_w30",
    "hrnetv2_w32",
    "hrnetv2_w40",
    "hrnetv2_w44",
    "hrnetv2_w48",
    "hrnetv2_w64",
]

variants_timm = [
    "hrnet_w18_small",
    "hrnet_w18_small_v2",
    "hrnet_w18",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
]


@pytest.mark.parametrize("variant", variants_osmr, ids=variants_osmr)
@pytest.mark.hrnet
def test_hrnet_osmr_pytorch(clear_pybuda, variant):
    run_hrnet_osmr_pytorch(variant)


@pytest.mark.parametrize("variant", variants_timm, ids=variants_timm)
@pytest.mark.hrnet
def test_hrnet_timm_pytorch(clear_pybuda, variant):
    run_hrnet_timm_pytorch(variant)
