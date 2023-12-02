import pytest

from cv_demos.mobilenet_v2.pytorch_mobilenet_v2_basic import run_mobilenetv2_basic
from cv_demos.mobilenet_v2.pytorch_mobilenet_v2_hf import run_mobilenetv2_hf
from cv_demos.mobilenet_v2.pytorch_mobilenet_v2_timm import run_mobilenetv2_timm

variants = [
    "google/mobilenet_v2_0.35_96",
    "google/mobilenet_v2_0.75_160",
    "google/mobilenet_v2_1.0_224",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.mobilenetv2
def test_mobilenetv2_hf_pytorch(clear_pybuda, variant):
    run_mobilenetv2_hf(variant)


@pytest.mark.mobilenetv2
def test_mobilenetv2_basic_pytorch(clear_pybuda):
    run_mobilenetv2_basic()


@pytest.mark.mobilenetv2
def test_mobilenetv2_timm_pytorch(clear_pybuda):
    run_mobilenetv2_timm()
