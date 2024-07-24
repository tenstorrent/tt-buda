# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Calculation of input shapes from output shapes for the specified operator


from typing import List

from .datatypes import TensorShape
from .datatypes import ShapeCalculationContext

from .datatypes import RandomizerTestContext

from .utils import RandomUtils


class OperatorShapes:

    @staticmethod
    def same_input_shapes(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        input_num, output_shape = calculation_context.input_num, calculation_context.output_shape
        # each input operand has the same shape as the output
        return [output_shape for _ in range(input_num)]

    @staticmethod
    def linear_inputs(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        output_shape = calculation_context.output_shape
        test_context: RandomizerTestContext = calculation_context.test_context
        # linear layer changes the last dimension of the input shape
        batch_shape = output_shape[:-1]
        n = output_shape[-1]
        n = randomize_size(len(batch_shape), test_context)
        input_shapes = [batch_shape + (n,)]
        return input_shapes

    # FIXME: conv2d in PyTorch not working properly in all cases
    @staticmethod
    def conv2d_inputs(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        output_shape = calculation_context.output_shape
        test_context: RandomizerTestContext = calculation_context.test_context
        shape1 = output_shape[:1]
        shape2 = output_shape[2:]
        n = output_shape[1]
        n = randomize_size(len(shape1), test_context)
        input_shapes = [shape1 + (n,) + shape2]
        return input_shapes

    @staticmethod
    def matmul_inputs(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        output_shape = calculation_context.output_shape
        test_context: RandomizerTestContext = calculation_context.test_context
        batch_shape = output_shape[:-2]
        m = output_shape[-2]
        n = output_shape[-1]
        # calculates inner dimension based on one of output shape dimensions
        # dim is wrong for second operand
        q = randomize_size(len(batch_shape) + 1, test_context)
        input_shapes = [batch_shape + (m,q), batch_shape + (q,n)]
        return input_shapes


def randomize_size(dim: int, test_context: RandomizerTestContext) -> int:
    '''Randomize size of a new dimension based operand size range

    Args:
        dim (int): new dimension
        test_context: RandomizerTestContext

    Returns:
        int: random size of an dimension
    '''
    rng_shape = test_context.rng_shape
    randomizer_config = test_context.randomizer_config
    op_size_min = randomizer_config.op_size_per_dim_min
    op_size_max = randomizer_config.op_size_per_dim_max
    quantization = randomizer_config.op_size_quantization

    n = rng_shape.randint(op_size_min, op_size_max)
    n = RandomUtils.quantize(n, quantization)
    # logger.trace(f"Randomize size: dim = {dim}, quant = {quantization} -> {n}")

    return n


class AdjustParameters:
    '''Adjust parameters for operators based on output shape'''
    # TODO Introduce adjustment method in operator definition similar to calc_input_shapes

