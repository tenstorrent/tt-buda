// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fork_quantization_scales.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "passes/passes_utils.hpp"
#include "passes/commute_utils.hpp"
#include "reportify/reportify.hpp"

namespace tt::passes {

bool fork_quantization_scales(graphlib::Graph *graph) {
    bool updated_anything = false;
    bool attempt_update = true;
    while(attempt_update) {
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (not is_quantization_ops(op))
                continue;

            graphlib::Node *scale = graph->data_operands(op)[1];

            std::vector<graphlib::Node *> ancestors_to_check{scale};

            while (ancestors_to_check.size() > 0) {
                scale = ancestors_to_check.back();
                ancestors_to_check.pop_back();
                if (graph->data_users(scale).size() > 1) {
                    ancestors_to_check.clear();
                    break;
                }

                for (graphlib::Node *operand : graph->data_operands(scale)) {
                    ancestors_to_check.push_back(operand);
                }
            }

            if (graph->data_users(scale).size() > 1) {
                fork_subgraph(graph, scale);
                attempt_update = true;
                updated_anything = true;
                break;
            }
            
        }
    }
    return updated_anything;
}

}