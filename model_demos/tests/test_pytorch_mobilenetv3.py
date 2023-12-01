import pytest
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_large_basic import \
    run_mobilenetv3_large_basic
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_large_timm import \
    run_mobilenetv3_large_timm
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_small_basic import \
    run_mobilenetv3_small_basic
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_small_timm import \
    run_mobilenetv3_small_timm


@pytest.mark.mobilenetv3
def test_mobilenetv3_large_basic_pytorch(clear_pybuda):
    run_mobilenetv3_large_basic()


@pytest.mark.mobilenetv3
def test_mobilenetv3_small_basic_pytorch(clear_pybuda):
    run_mobilenetv3_small_basic()


@pytest.mark.mobilenetv3
def test_mobilenetv3_large_timm_pytorch(clear_pybuda):
    run_mobilenetv3_large_timm()


@pytest.mark.mobilenetv3
def test_mobilenetv3_small_timm_pytorch(clear_pybuda):
    run_mobilenetv3_small_timm()
