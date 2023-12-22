import pytest

from nlp_demos.fuyu8b.pytorch_fuyu8b_past_cache import run_fuyu8b_past_cache


@pytest.mark.fuyu8b
def test_fuyu8b_past_cache_pytorch(clear_pybuda):
    run_fuyu8b_past_cache()
