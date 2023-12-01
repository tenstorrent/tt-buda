import pytest
from cv_demos.vit.pytorch_vit_classify_16_224_hf import \
    run_vit_classify_224_hf_pytorch

variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.vit
def test_vit_classify_224_hf_pytorch(clear_pybuda, variant):
    run_vit_classify_224_hf_pytorch(variant)
