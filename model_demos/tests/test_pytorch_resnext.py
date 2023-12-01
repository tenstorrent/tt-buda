import pytest
from cv_demos.resnext.pytorch_resnext import run_resnext_pytorch

variants = [
    ("resnext14_32x4d", "osmr"),
    ("resnext26_32x4d", "osmr"),
    ("resnext50_32x4d", "osmr"),
    ("resnext101_64x4d", "osmr"),
    ("resnext50_32x4d", "pytorch/vision:v0.10.0"),
    ("resnext101_32x8d_wsl", "facebookresearch/WSL-Images"),
    ("resnext101_32x8d", "pytorch/vision:v0.10.0"),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.resnext
def test_resnext_pytorch(clear_pybuda, variant):
    run_resnext_pytorch(variant)
