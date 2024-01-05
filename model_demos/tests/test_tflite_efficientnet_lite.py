import pytest

from cv_demos.efficientnet_lite.tflite_efficientnet_lite0_1x1 import run_efficientnet_lite0_1x1
from cv_demos.efficientnet_lite.tflite_efficientnet_lite4_1x1 import run_efficientnet_lite4_1x1


@pytest.mark.efficientnetlite
def test_efficientnet_lite0_1x1(clear_pybuda):
    run_efficientnet_lite0_1x1()


@pytest.mark.efficientnetlite
def test_efficientnet_lite4_1x1(clear_pybuda):
    run_efficientnet_lite4_1x1()
