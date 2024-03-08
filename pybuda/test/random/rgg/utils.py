# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Utility functions


from typing import Dict
import torch
import re

import numpy as np

from pybuda.op_repo import OperatorParam, OperatorParamNumber

from .datatypes import RandomizerTestContext


# TODO move to StrUtils
def tensor_shape(t: torch.Tensor) -> str:
    """
    Returns a string representation of the shape of a tensor.
    The method is used for logging purposes.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        str: A string representation of the tensor shape, in the format "dim=<number of dimensions> v=<size of each dimension>".
    """
    return f"dim={t.dim()} v={t.size()}"


class StrUtils:

    @staticmethod
    def kwargs_str(**kwargs):
        s = ', '.join([f"{key}= {value}" for key, value in kwargs.items()])
        if s:
            s = ", " + s
        return s

    @staticmethod
    def args_str(*args):
        s = ', '.join([f"{value}" for value in args])
        if s:
            s = ", " + s
        return s

    @staticmethod
    def camel_case_to_snake_case(camel_case: str) -> str:
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        snake_case = re.sub(pattern, '_', camel_case).lower()
        return snake_case

    @classmethod
    def test_id(cls, test_context: RandomizerTestContext) -> str:
        parameters = test_context.parameters
        graph_builder_snake_case = cls.camel_case_to_snake_case(parameters.graph_builder_name)
        test_id = f"{parameters.framework_name}_{graph_builder_snake_case}_{parameters.test_index}_{parameters.random_seed}"
        return test_id


class RandomUtils:

    @classmethod
    def random_value_for_param(cls, param: OperatorParam):
        if isinstance(param, OperatorParamNumber):
            return cls.random_value_for_number_param(param)
        else:
            raise ValueError(f"Unsupported param type {type(param)}")

    @classmethod
    def random_value_for_number_param(cls, param: OperatorParamNumber) -> int:
        # TODO: reuse seed
        # TODO: support open intervals
        if param.type == float:
            return np.random.uniform(param.min_value, param.max_value)
        elif param.type == int:
            return np.random.randint(param.min_value, param.max_value + 1)
        else:
            raise ValueError(f"Unsupported type {param.type}")

    @classmethod
    def constructor_kwargs(cls, param: OperatorParam) -> Dict:
        return {param.name: cls.random_value_for_param(param) for param in param.constructor_params}

    @classmethod
    def forward_kwargs(cls, param: OperatorParam) -> Dict:
        return {param.name: cls.random_value_for_param(param) for param in param.forward_params}
