# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Falcon-7B finetune tests

import os
import pytest
from .falcon_modules.falcon import run_finetune

# Requires around 14 GB for falcon7B model (tiiuae/falcon-7b)
os.environ['HF_DATASETS_CACHE']='/proj_sw/large-model-cache/falcon7b'
os.environ['TRANSFORMERS_CACHE']='/proj_sw/large-model-cache/falcon7b'

# Tests Falcon-7B finetune basic demo
# Config 1 (ci_basic.json): basic test with wq/wv lora modules
# Config 2 (ci_basic_lora.json): basic test with all lora modules, rank 2, precision low-mp
@pytest.mark.parametrize("config_file", [('pybuda/test/falcon/finetune_configs/ci_basic.json'), ('pybuda/test/falcon/finetune_configs/ci_basic_lora.json')])
def test_finetune_basic(config_file):
    run_finetune(config_file)
