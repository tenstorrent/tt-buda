import pytest

from cv_demos.xception.timm_xception import run_xception_timm

variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xception
def test_xception_timm_pytorch(clear_pybuda, variant):
    run_xception_timm(variant)
