import pytest
from cv_demos.mobilenet_v1.pytorch_mobilenet_v1_basic import \
    run_mobilenetv1_basic
from cv_demos.mobilenet_v1.pytorch_mobilenet_v1_hf import run_mobilenetv1_hf

variants = [
    "google/mobilenet_v1_0.75_192",
    "google/mobilenet_v1_1.0_224",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.mobilenetv1
def test_mobilenetv1_hf_pytorch(clear_pybuda, variant):
    run_mobilenetv1_hf(variant)


@pytest.mark.mobilenetv1
def test_mobilenetv1_basic_pytorch(clear_pybuda):
    run_mobilenetv1_basic()
