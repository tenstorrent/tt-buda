# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Test random graph configurations by utilizing Random Graph Generator Algorithm and targeting PyBuda and PyTorch frameworks

import pytest

from test.random.rgg import Frameworks
from test.random.rgg import RandomGraphAlgorithm
from test.random.rgg import process_test


# @pytest.mark.parametrize("framework", [framework.value for framework in Frameworks])
@pytest.mark.parametrize("framework", [
    Frameworks.PYBUDA.value,
    Frameworks.PYTORCH.value,
])
def test_random_graph_algorithm(test_index, random_seeds, test_device, randomizer_config, framework):
    # TODO random_seed instead of random_seeds
    random_seed = random_seeds[test_index]
    process_test(test_index, random_seed, test_device, randomizer_config, graph_builder_type=RandomGraphAlgorithm, framework=framework)
