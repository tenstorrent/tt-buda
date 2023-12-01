import pytest

from cv_demos.clip.pytorch_clip import run_clip_pytorch

variants = ["openai/clip-vit-base-patch32"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.clip
def test_clip_pytorch(clear_pybuda, variant):
    run_clip_pytorch(variant)
