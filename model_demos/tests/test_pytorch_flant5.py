import pytest
from nlp_demos.flan_t5.pytorch_flan_t5_generation import \
    run_flan_t5_pybuda_pipeline

variants = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.flant5
def test_flan_t5_generation_pytorch(clear_pybuda, variant):
    run_flan_t5_pybuda_pipeline(variant)
