import pytest

from cv_demos.conv_autoencoder.pytorch_conv_autoencoder import run_conv_ae_pytorch
from cv_demos.linear_autoencoder.pytorch_linear_autoencoder import run_linear_ae_pytorch


@pytest.mark.autoencoder
def test_linear_ae_pytorch(clear_pybuda):
    run_linear_ae_pytorch()


@pytest.mark.autoencoder
def test_conv_ae_pytorch(clear_pybuda):
    run_conv_ae_pytorch()
