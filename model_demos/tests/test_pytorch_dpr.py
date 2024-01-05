import pytest

from nlp_demos.dpr.pytorch_dpr_context_encoder import run_dpr_context_encoder_pytorch
from nlp_demos.dpr.pytorch_dpr_question_encoder import run_dpr_question_encoder_pytorch
from nlp_demos.dpr.pytorch_dpr_reader import run_dpr_reader_pytorch

variants_ctx = ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]
variants_qe = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]
variants_reader = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]


@pytest.mark.parametrize("variant", variants_ctx, ids=variants_ctx)
@pytest.mark.dpr
def test_dpr_context_encoder_pytorch(clear_pybuda, variant):
    run_dpr_context_encoder_pytorch(variant)


@pytest.mark.parametrize("variant", variants_qe, ids=variants_qe)
@pytest.mark.dpr
def test_dpr_question_encoder_pytorch(clear_pybuda, variant):
    run_dpr_question_encoder_pytorch(variant)


@pytest.mark.parametrize("variant", variants_reader, ids=variants_reader)
@pytest.mark.dpr
def test_dpr_reader_pytorch(clear_pybuda, variant):
    run_dpr_reader_pytorch(variant)
