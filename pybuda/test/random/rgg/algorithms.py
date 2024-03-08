# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Implementation of randomization algrorithms


from loguru import logger
import random
from .base import RandomizerNode, RandomizerGraph, GraphBuilder
from .base import Framework


class RandomGraphAlgorithm(GraphBuilder):

    def __init__(self, framework: Framework, randomizer_config):
        super(RandomGraphAlgorithm, self).__init__(randomizer_config)
        self.framework = framework
        self.operators = [op for op in framework.operator_repository.operators if op.input_num == 1 and not op.is_layer()]

    def get_single_input_operator(self, rng):
        return rng.choice(self.operators)

    def build_graph(self, parameters):
        rng = random.Random(parameters.random_seed)

        min_op_size = self.randomizer_config.min_op_size
        max_op_size = self.randomizer_config.max_op_size
        num_of_nodes = self.randomizer_config.num_of_nodes
        
        rows = rng.randint(min_op_size, max_op_size)
        cols1 = rng.randint(min_op_size, max_op_size)
        cols2 = rng.randint(min_op_size, max_op_size)
        microbatch_size = rng.randint(1, 8)

        graph = RandomizerGraph([])
        nodes = graph.nodes

        for _ in range(rng.randint(1, num_of_nodes)):
            # Choose operator randomly based on rng
            op1 = self.get_single_input_operator(rng)

            # if op1.is_layer:
            #     # Layers require number of input and output features
            #     nodes.append(RandomizerNode(operator=op1, in_features=cols1, out_features=cols2))
            # else:
            #     nodes.append(RandomizerNode(operator=op1))
            nodes.insert(0, RandomizerNode(operator=op1))

        # Provide input shape for validation
        graph.input_shape = (microbatch_size, rows, cols1)

        logger.debug(f"rows={rows} cols1={cols1} cols2={cols2} microbatch_size={microbatch_size} input_shape={graph.input_shape}")

        return graph
