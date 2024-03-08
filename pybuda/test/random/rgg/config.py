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
        min_op_size=int(os.environ.get("MIN_OP_SIZE", 16)),
        max_op_size=int(os.environ.get("MAX_OP_SIZE", 512)),
        num_of_nodes=int(os.environ.get("NUM_OF_NODES", 10)),
    )
    return randomizer_config
