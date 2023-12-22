import pytest

from cv_demos.vilt.pytorch_vilt_maskedlm import run_vilt_maskedlm_pytorch
from cv_demos.vilt.pytorch_vilt_question_answering import run_vilt_for_question_answering_pytorch


@pytest.mark.vilt
def test_vilt_for_question_answering_pytorch(clear_pybuda):
    run_vilt_for_question_answering_pytorch()


@pytest.mark.vilt
def test_vilt_maskedlm_pytorch(clear_pybuda):
    run_vilt_maskedlm_pytorch()
