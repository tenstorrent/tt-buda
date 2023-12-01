import pytest

from cv_demos.unet.pytorch_unet_qubvel import run_unet_qubvel_pytorch
from cv_demos.unet.pytorch_unet_torchhub import run_unet_torchhub_pytorch


@pytest.mark.unet
def test_unet_qubvel(clear_pybuda):
    run_unet_qubvel_pytorch()


@pytest.mark.unet
def test_unet_torchhub(clear_pybuda):
    run_unet_torchhub_pytorch()
