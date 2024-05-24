# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Void op is for testing purposes only

import torch
import pybuda


def eval(type, attr, ops):
    return torch.zeros(1, 1, 32, 32)


def shape(type, attr, ops, tile_height, tile_width):
    return [1, 1, 32, 32], []


def parallelization(type, attr, op_shape):
    raise RuntimeError("parallelization intentionally not defined for void op")


def input_ublock_order(type, attr, num_operands):
    raise RuntimeError("input_ublock_order intentionally not defined for void op")


def execution_cycles(type, arch_name, op_model) -> int:
    raise RuntimeError("execution_cycles intentionally not defined for void op")
