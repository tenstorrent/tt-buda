import pytest
from nlp_demos.squeezebert.pytorch_squeezebert_sequence_classification import \
    run_squeezebert_sequence_classification_pytorch


@pytest.mark.squeezebert
def test_squeezebert_sequence_classification_pytorch(clear_pybuda):
    run_squeezebert_sequence_classification_pytorch()
