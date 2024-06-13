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
        # debug_forward = True,
        run_test = True,
        save_tests = True,
        # build_model_from_code = False,
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
        num_of_nodes=int(os.environ.get("NUM_OF_NODES", 10)),
    )
    return randomizer_config
