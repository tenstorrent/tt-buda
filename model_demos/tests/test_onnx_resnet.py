import pytest

from cv_demos.resnet.onnx_resnet import run_resnet_onnx


@pytest.mark.resnet
def test_resnet_onnx(clear_pybuda):
    run_resnet_onnx()
