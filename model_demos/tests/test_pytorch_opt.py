import pytest
from nlp_demos.opt.pytorch_opt_causal_lm import run_opt_casual_lm
from nlp_demos.opt.pytorch_opt_question_answering import \
    run_opt_question_answering
from nlp_demos.opt.pytorch_opt_sequence_classification import \
    run_opt_sequence_classification

variants = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.opt
def test_opt_causal_lm_pytorch(clear_pybuda, variant):
    run_opt_casual_lm(variant)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.opt
def test_opt_question_answering_pytorch(clear_pybuda, variant):
    run_opt_question_answering(variant)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.opt
def test_opt_sequence_classification_pytorch(clear_pybuda, variant):
    run_opt_sequence_classification(variant)
