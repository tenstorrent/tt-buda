# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Configuration of the randomizer


import os

from .datatypes import RandomizerConfig


# TODO introduce environment variables to set these values
# TODO read config from file
def get_randomizer_config_default():
    randomizer_config = RandomizerConfig (
        print_graph = False,
        print_code = True,
        run_test = True,
        save_tests = True,
        save_failing_tests = True,
        # build_model_from_code = False,
        debug_shapes = False,
        verify_shapes = False,
        # TODO ranges
        # dim_min=int(os.environ.get("MIN_DIM", 3)),
        dim_min=int(os.environ.get("MIN_DIM", 4)),  # Until #2722 is resolved
        dim_max=int(os.environ.get("MAX_DIM", 4)),
        op_size_per_dim_min=int(os.environ.get("MIN_OP_SIZE_PER_DIM", 16)),
        op_size_per_dim_max=int(os.environ.get("MAX_OP_SIZE_PER_DIM", 64)),  # by default run with smaller sizes
        # op_size_per_dim_max=int(os.environ.get("MAX_OP_SIZE_PER_DIM", 512)),
        microbatch_size_min=int(os.environ.get("MIN_MICROBATCH_SIZE", 1)),
        microbatch_size_max=int(os.environ.get("MAX_MICROBATCH_SIZE", 8)),
        num_of_nodes_min=int(os.environ.get("NUM_OF_NODES_MIN", 5)),
        num_of_nodes_max=int(os.environ.get("NUM_OF_NODES_MAX", 10)),
        num_fork_joins_max=int(os.environ.get("NUM_OF_FORK_JOINS_MAX", 50)),
        constant_input_rate=int(os.environ.get("CONSTANT_INPUT_RATE", 20)),
        same_inputs_percent_limit=int(os.environ.get("SAME_INPUTS_PERCENT_LIMIT", 10)),
    )
    return randomizer_config
