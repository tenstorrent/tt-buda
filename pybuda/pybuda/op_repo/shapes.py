# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Calculation of input shapes from output shapes for the specified operator


from random import Random
from typing import List

from .datatypes import OperatorDefinition
from .datatypes import TensorShape


def same_input_shapes(operator_definition: OperatorDefinition, output_shape: TensorShape, rng_shape: Random) -> List[TensorShape]:
    # each input operand has the same shape as the output
    return [output_shape for _ in range(operator_definition.input_num)]


def linear_inputs(operator_definition: OperatorDefinition, output_shape: TensorShape, rng_shape: Random) -> List[TensorShape]:
    # linear layer changes the last dimension of the input shape
    batch_shape = output_shape[:-1]
    n = output_shape[-1]
    n = randomize_size(n, rng_shape)
    input_shapes = [batch_shape + (n,)]
    return input_shapes


# FIXME: conv2d in PyTorch not working properly in all cases
def conv2d_inputs(operator_definition: OperatorDefinition, output_shape: TensorShape, rng_shape: Random) -> List[TensorShape]:
    shape1 = output_shape[:1]
    shape2 = output_shape[2:]
    n = output_shape[1]
    n = randomize_size(n, rng_shape)
    input_shapes = [shape1 + (n,) + shape2]
    return input_shapes


def matmul_inputs(operator_definition: OperatorDefinition, output_shape: TensorShape, rng_shape: Random) -> List[TensorShape]:
    batch_shape = output_shape[:-2]
    m = output_shape[-2]
    n = output_shape[-1]
    # calculates inner dimension based on one of output shape dimensions
    q = randomize_size(n, rng_shape)
    input_shapes = [batch_shape + (m,q), batch_shape + (q,n)]
    return input_shapes


def randomize_size(n: int, rng_shape: Random) -> int:
    '''Randomize size of an dimension based on size of another dimension.
    Returns a random integer in the range [n/2, 3n/2] inclusive to keep the size of the dimension in a similar range.

    Args:
        n: size of an dimension
        rng_shape: random number generator

    Returns:
        int: random size of an dimension
    '''
    return n + (rng_shape.randint(0, 1) * 2 - 1) * rng_shape.randint(0, n // 2)
