import pytest
from nlp_demos.bert.pytorch_bert_masked_lm import run_bert_masked_lm_pytorch
from nlp_demos.bert.pytorch_bert_question_answering import \
    run_bert_question_answering_pytorch
from nlp_demos.bert.pytorch_bert_sequence_classification import \
    run_bert_sequence_classification_pytorch
from nlp_demos.bert.pytorch_bert_token_classification import \
    run_bert_token_classification_pytorch


@pytest.mark.bert
def test_bert_masked_lm_pytorch(clear_pybuda):
    run_bert_masked_lm_pytorch()


@pytest.mark.bert
def test_bert_question_answering_pytorch(clear_pybuda):
    run_bert_question_answering_pytorch()


@pytest.mark.bert
def test_bert_sequence_classification_pytorch(clear_pybuda):
    run_bert_sequence_classification_pytorch()


@pytest.mark.bert
def test_bert_token_classification_pytorch(clear_pybuda):
    run_bert_token_classification_pytorch()
