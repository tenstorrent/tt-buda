import pytest
from nlp_demos.roberta.pytorch_roberta_masked_lm import run_roberta_mlm_pytorch
from nlp_demos.roberta.pytorch_roberta_sentiment import \
    run_roberta_sentiment_pytorch


@pytest.mark.roberta
def test_roberta_mlm_pytorch(clear_pybuda):
    run_roberta_mlm_pytorch()


@pytest.mark.roberta
def test_roberta_sentiment_pytorch(clear_pybuda):
    run_roberta_sentiment_pytorch()
