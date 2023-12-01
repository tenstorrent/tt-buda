import pytest
from nlp_demos.falcon.pytorch_falcon import run_falcon_pytorch


@pytest.mark.falcon
def test_falcon_pytorch(clear_pybuda):
    run_falcon_pytorch()
