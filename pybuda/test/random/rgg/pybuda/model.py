# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Building PyBuda models


from typing import Type
from loguru import logger

from pybuda import PyBudaModule

from .. import RandomizerGraph, ModelBuilder, StrUtils


class PyBudaModelBuilder(ModelBuilder):

    def build_model(self, graph: RandomizerGraph, GeneratedTestModel: Type[PyBudaModule]) -> PyBudaModule:
        module_name = f"gen_model_pytest_{StrUtils.test_id(graph)}"
        pybuda_model = GeneratedTestModel(module_name)
        return pybuda_model
