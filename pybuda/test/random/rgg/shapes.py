# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Calculation of input shapes from output shapes for the specified operator


import random

from loguru import logger
from typing import List

from .datatypes import TensorShape
from .datatypes import RandomizerNode
from .datatypes import InvalidShape
from .datatypes import RandomizerTestContext
from .datatypes import ShapeCalculationContext

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

    @staticmethod
    def interleave_inputs(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        # Interleave joins the input shapes along the specified axis
        # It requires that axis dimension is divisible by the number of inputs
        input_num, output_shape = calculation_context.input_num, calculation_context.output_shape
        forward_kwargs = calculation_context.forward_kwargs
        axis = forward_kwargs["axis"]

        if axis >= len(output_shape) or axis < 0:
            axis %= len(output_shape)

        logger.trace(f"Interleave axis = {axis} output_shape = {output_shape}")

        shape1 = output_shape[:axis]
        mid_size = output_shape[axis]
        shape2 = output_shape[axis+1:]

        if mid_size < input_num:
            raise InvalidShape(f"Output shape {output_shape} is too small mid_size={mid_size} < input_num={input_num}")

        if mid_size % input_num != 0:
            raise InvalidShape(f"Output shape {output_shape} axis[{axis}]={mid_size} is not divisible by input_num={input_num}")

        dim = mid_size // input_num

        input_shapes = [shape1 + (dim,) + shape2 for _ in range(input_num)]
        return input_shapes

    @staticmethod
    def concatenate_inputs(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        # Concatenate joins the input shapes along the specified axis
        # It requires that axis dimension can be split into input_num parts
        input_num, output_shape = calculation_context.input_num, calculation_context.output_shape
        test_context: RandomizerTestContext = calculation_context.test_context
        rng_shape = test_context.rng_shape
        forward_kwargs = calculation_context.forward_kwargs
        axis = forward_kwargs["axis"]

        if axis >= len(output_shape) or axis < 0:
            axis %= len(output_shape)

        logger.trace(f"Concatenate axis = {axis} output_shape = {output_shape}")

        shape1 = output_shape[:axis]
        mid_size = output_shape[axis]
        shape2 = output_shape[axis+1:]

        if mid_size < input_num:
            raise InvalidShape(f"Output shape {output_shape} is too small mid_size={mid_size} < input_num={input_num}")

        dims = []
        for input_pos in range(input_num):
            reserved_size = input_num - input_pos - 1
            mid_range = mid_size - reserved_size
            logger.trace(f"input_num = {input_num} mid_size = {mid_size} reserved_size = {reserved_size} mid_range = {mid_range}")
            if mid_range <= 0:
                raise InvalidShape(f"Output shape {output_shape} is too small mid_range={mid_range} <= 0")
            if reserved_size == 0:
                dim = mid_size
            else:
                # TODO quantize size
                dim = rng_shape.randint(1, mid_range)
            logger.trace(f"dim = {dim}")
            mid_size -= dim
            dims.append(dim)

        input_shapes = [shape1 + (dim,) + shape2 for dim in dims]
        return input_shapes

    @staticmethod
    def stack_inputs(calculation_context: ShapeCalculationContext) -> List[TensorShape]:
        # Stack adds a new dimension at the specified axis
        input_num, output_shape = calculation_context.input_num, calculation_context.output_shape
        test_context: RandomizerTestContext = calculation_context.test_context
        forward_kwargs = calculation_context.forward_kwargs
        axis = forward_kwargs["axis"]

        if len(output_shape) <= test_context.randomizer_config.dim_min:
            raise InvalidShape(f"Output shape {output_shape} is too small len(output_shape)={len(output_shape)} <= dim_min={test_context.randomizer_config.dim_min}")
        dim = output_shape[axis]
        if dim != input_num:
            raise InvalidShape(f"Mismatch of dim and input_num in output shape {output_shape}. dim={dim} != input_num={input_num}")
        shape1 = output_shape[:axis]
        shape2 = output_shape[axis+1:]

        input_shapes = [shape1 + shape2 for _ in range(input_num)]
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

    @staticmethod
    def interleave_adjust(node: RandomizerNode, test_context: RandomizerTestContext) -> None:
        '''Adjust parameters and input number for interleave based on output shape'''
        rng_shape = test_context.rng_shape
        input_num_range = node.operator.input_num_range

        input_num = node.input_num
        output_shape = node.output_shape
        axis = node.forward_kwargs["axis"]

        if len(output_shape) < 4:
            raise InvalidShape(f"Output shape {node.output_shape} has len(output_shape)={len(output_shape)} < 4")

        if axis != -3:
            raise InvalidShape(f"Invalid axis={axis} for output shape {node.output_shape}")

        mid_size = output_shape[axis]

        logger.trace(f"Interleave axis = {axis} output_shape = {output_shape} mid_size = {mid_size} input_num = {input_num}")

        if mid_size % input_num == 0:
            # If axis is divisible by input number, no need to recalculate
            return

        # Currently axis is required to be -3 so no need to change axis
        supported_axises = [(axis, node.output_shape[axis])]
        # supported_axises = list(enumerate(node.output_shape))

        for axis, mid_size in rng_shape.sample(supported_axises, len(supported_axises)):
            for input_num in rng_shape.sample(range(input_num_range.operands_min, input_num_range.operands_max+1), input_num_range.operands_max - input_num_range.operands_min + 1):
                if mid_size % input_num == 0:
                    node.forward_kwargs["axis"] = axis
                    node.input_num = input_num
                    node.init_inputs()
                    return

        raise InvalidShape(f"Not found possible params for output shape {node.output_shape}")

    @staticmethod
    def concatenate_adjust(node: RandomizerNode, test_context: RandomizerTestContext) -> None:
        '''Adjust parameters and input number for concatenate based on output shape'''
        rng_shape = test_context.rng_shape
        input_num_range = node.operator.input_num_range

        input_num = node.input_num
        output_shape = node.output_shape
        axis = node.forward_kwargs["axis"]

        if not -len(output_shape) <= axis < len(output_shape):
            axis = None  # must be recalculated

        if axis is not None and axis % len(output_shape) == 0:
            # Axis 0 is not supported
            axis = None  # must be recalculated

        if axis is not None:
            # Maybe it's possible axis
            axis %= len(output_shape)

            mid_size = output_shape[axis]

            if mid_size >= input_num:
                # It is possible axis, no need to recalculate
                return

            # TODO global limit for number of operands
            if input_num_range.operands_min <= mid_size <= input_num_range.operands_max:
                # Axis is possible but number of inputs is too big
                # Lower number of inputs to fit axis dimension
                node.input_num = rng_shape.randint(input_num_range.operands_min, mid_size)
                node.init_inputs()
                return

        # Try another axis
        for axis, mid_size in rng_shape.sample(list(enumerate(node.output_shape)), len(node.output_shape)):
            if axis % len(output_shape) == 0:
                # Axis 0 is not supported
                continue
            if input_num_range.operands_min <= mid_size:
                node.forward_kwargs["axis"] = axis
                node.input_num = rng_shape.randint(input_num_range.operands_min, min(mid_size, input_num_range.operands_max))
                node.init_inputs()
                return
        
        raise InvalidShape(f"Not found possible params for output shape {node.output_shape}")

    @staticmethod
    def stack_adjust(node: RandomizerNode, test_context: RandomizerTestContext) -> None:
        '''Adjust parameters and input number for stack based on output shape'''
        input_num_range = node.operator.input_num_range
        output_shape = node.output_shape
        if len(output_shape) <= test_context.randomizer_config.dim_min:
            raise InvalidShape(f"Output shape {output_shape} is too small len(output_shape)={len(output_shape)} <= dim_min={test_context.randomizer_config.dim_min}")
        for axis, dim in enumerate(node.output_shape):
            if axis == 0:
                # Axis 0 is not supported
                continue
            if input_num_range.operands_min <= dim <= input_num_range.operands_max:
                node.forward_kwargs["axis"] = axis
                node.input_num = dim
                node.init_inputs()
                return
        raise InvalidShape(f"Not found possible params for output shape {node.output_shape}")
