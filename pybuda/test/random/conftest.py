# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import random
import os
import pybuda

test_rg = random.Random()
seeds = []
    
@pytest.fixture(autouse=True)
def run_test(test_index, random_seeds):
    pybuda.config.set_configuration_options(balancer_policy="Random", use_interactive_placer=True)
    os.environ["PYBUDA_BALANCER_RANDOM_POLICY_SEED"] = str(random_seeds[test_index])
    
    rng = random.Random(random_seeds[test_index])
    
    # Pick a random data format, bfp8 and up
    df = rng.choice([pybuda.DataFormat.Bfp8_b, pybuda.DataFormat.Float16_b, pybuda.DataFormat.Float16, pybuda.DataFormat.Float32])
    pybuda.config.set_configuration_options(default_df_override=df)

    # Enable AMP
    amp = rng.choice([0, 1, 2])
    pybuda.config.set_configuration_options(amp_level=amp)

    yield

def pytest_generate_tests(metafunc):
    if "test_index" in metafunc.fixturenames:
        if "RANDOM_TEST_COUNT" in os.environ:
            test_count = int(os.environ["RANDOM_TEST_COUNT"])
        else:
            test_count = 5
        tests_selected_indecies = []
        if "RANDOM_TESTS_SELECTED" in os.environ:
            tests_selected = os.environ["RANDOM_TESTS_SELECTED"]
            tests_selected = tests_selected.strip()
            if len(tests_selected) > 0:
                tests_selected_indecies = tests_selected.split(",")
                tests_selected_indecies = [int(i) for i in tests_selected_indecies]
        if len(tests_selected_indecies) > 0:
            metafunc.parametrize("test_index", tests_selected_indecies)
            last_test_selected = max(tests_selected_indecies)
            if test_count < last_test_selected + 1:
                test_count = last_test_selected + 1
        else:
            metafunc.parametrize("test_index", range(test_count))

        global seeds
        if len(seeds) > 0:
            return 

        if "RANDOM_TEST_SEED" in os.environ:
            test_rg.seed(int(os.environ["RANDOM_TEST_SEED"]))
        else:
            test_rg.seed(0)

        seeds = []
        # generate a new random seed for each test. Do it upfront so that
        # we can run any index in isolation
        for _ in range(test_count):
            seeds.append(test_rg.randint(0, 1000000))

@pytest.fixture
def random_seeds():
    return seeds


