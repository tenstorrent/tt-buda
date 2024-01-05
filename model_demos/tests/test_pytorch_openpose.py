import pytest

from cv_demos.openpose.pytorch_lwopenpose_2d_osmr import run_lwopenpose_2d_osmr_pytorch
from cv_demos.openpose.pytorch_lwopenpose_3d_osmr import run_lwopenpose_3d_osmr_pytorch


@pytest.mark.openpose
def test_openpose_2d_osmr(clear_pybuda):
    run_lwopenpose_2d_osmr_pytorch()


@pytest.mark.openpose
def test_openpose_3d_osmr(clear_pybuda):
    run_lwopenpose_3d_osmr_pytorch()
