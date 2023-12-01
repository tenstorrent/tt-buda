import pytest

from cv_demos.googlenet.pytorch_googlenet_torchhub import run_googlenet_pytorch


@pytest.mark.googlenet
def test_googlenet_pytorch(clear_pybuda):
    run_googlenet_pytorch()
