import pytest

from cv_demos.retinanet.onnx_retinanet_r101 import run_retinanet_r101_640x480_onnx


@pytest.mark.retinanet
def test_retinanet_onnx(clear_pybuda):
    run_retinanet_r101_640x480_onnx()
