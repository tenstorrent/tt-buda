import pytest
from cv_demos.densenet.pytorch_densenet import run_densenet_pytorch
from cv_demos.densenet.pytorch_densenet_121_hf_xray import \
    run_densenet_121_hf_xray_pytorch

variants = ["densenet121", "densenet161", "densenet169", "densenet201"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.densenet
def test_densenet_pytorch(clear_pybuda, variant):
    run_densenet_pytorch(variant)


@pytest.mark.densenet
def test_densenet_121_hf_xray_pytorch(clear_pybuda):
    run_densenet_121_hf_xray_pytorch()
