import pytest
from nlp_demos.gpt2.gpt2_text_generation import run_gpt2_text_gen


@pytest.mark.gpt2
def test_gpt2_pytorch(clear_pybuda):
    run_gpt2_text_gen()
