import pytest
from nlp_demos.xglm.pytorch_xglm_causal_lm import run_xglm_causal_lm

variants = ["facebook/xglm-564M", "facebook/xglm-1.7B"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xglm
def test_xglm_causal_lm_pytorch(clear_pybuda, variant):
    run_xglm_causal_lm(variant)
