import pytest

from cv_demos.yolo_v3.pytorch_yolov3_holli import run_yolov3_holli_pytorch
from cv_demos.yolo_v3.pytorch_yolov3_holli_1x1 import run_yolov3_holli_pytorch_1x1
from cv_demos.yolo_v3.pytorch_yolov3_tiny_holli import run_yolov3_tiny_holli_pytorch


@pytest.mark.yolov3
def test_yolov3_holli(clear_pybuda):
    run_yolov3_holli_pytorch()


@pytest.mark.yolov3
def test_yolov3_holli_tiny(clear_pybuda):
    run_yolov3_tiny_holli_pytorch()


@pytest.mark.yolov3
def test_yolov3_holli_1x1(clear_pybuda):
    run_yolov3_holli_pytorch_1x1()
