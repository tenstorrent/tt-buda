import pytest
from nlp_demos.codegen.pytorch_codegen_causal_lm import run_codegen_causal_lm

# Variants: Salesforce/codegen-350M-mono, Salesforce/codegen-350M-multi, Salesforce/codegen-350M-nl
# Salesforce/codegen-2B-mono, Salesforce/codegen-2B-multi, Salesforce/codegen-2B-nl
# Note: only 1 variant selected for continuous testing, change as needed.
variants = [
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-nl",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.codegen
def test_codegen_causal_lm(clear_pybuda, variant):
    run_codegen_causal_lm(variant)
