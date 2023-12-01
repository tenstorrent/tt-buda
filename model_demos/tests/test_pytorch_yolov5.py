import pytest

from cv_demos.yolo_v5.pytorch_yolov5_320 import run_pytorch_yolov5_320
from cv_demos.yolo_v5.pytorch_yolov5_480 import run_pytorch_yolov5_480
from cv_demos.yolo_v5.pytorch_yolov5_640 import run_pytorch_yolov5_640

variants = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolov5
def test_pytorch_yolov5_320(clear_pybuda, variant):
    run_pytorch_yolov5_320(variant)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolov5
def test_pytorch_yolov5_640(clear_pybuda, variant):
    run_pytorch_yolov5_640(variant)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolov5
def test_pytorch_yolov5_480(clear_pybuda, variant):
    run_pytorch_yolov5_480(variant)
