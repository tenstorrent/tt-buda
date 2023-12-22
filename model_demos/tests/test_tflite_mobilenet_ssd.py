import pytest

from model_demos.cv_demos.mobilenet_ssd.tflite_mobilenet_v2_ssd_1x1 import run_mobilenetv2_ssd_1x1_tflite


@pytest.mark.mobilenetssd
def test_mobilenetv2_ssd_1x1_tflite(clear_pybuda):
    run_mobilenetv2_ssd_1x1_tflite()
