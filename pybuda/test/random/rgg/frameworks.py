# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# In depth testing of PyBuda models with one randomly selected operation


from enum import Enum

from .base import Framework

from .pybuda.model import PyBudaModelBuilder
from .pytorch.model import PyTorchModelBuilder

from pybuda.op_repo import pybuda_operator_repository
from pybuda.op_repo import pytorch_operator_repository


class Frameworks(Enum):
    ''' Register of all frameworks '''

    PYBUDA = Framework(
        framework_name="PyBuda",
        ModelBuilderType=PyBudaModelBuilder,
        operator_repository=pybuda_operator_repository,
    )
    PYTORCH = Framework(
        framework_name="PyTorch",
        ModelBuilderType=PyTorchModelBuilder,
        operator_repository=pytorch_operator_repository,
    )
