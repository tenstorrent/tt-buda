import pytest

from cv_demos.mlpmixer.timm_mlpmixer import run_mlpmixer_timm


@pytest.mark.mlpmixer
def test_mlpmixer_timm(clear_pybuda):
    run_mlpmixer_timm()
