import pytest

from cv_demos.stable_diffusion.pytorch_stable_diffusion import run_stable_diffusion_pytorch


@pytest.mark.stablediffusion
def test_stable_diffusion_pytorch(clear_pybuda):
    run_stable_diffusion_pytorch()
