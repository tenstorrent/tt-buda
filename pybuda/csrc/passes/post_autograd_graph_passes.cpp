// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "post_autograd_graph_passes.hpp"
#include "utils/logger.hpp"

using tt::LogAutograd;

namespace tt {

void lower_bwd_gather_ops(Graph *graph) {
    auto is_gather_op = [](graphlib::Node *n) -> bool {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(n);
        return op and op->op_name() == "gather";
    };

    auto is_gather_collapse = [graph, is_gather_op](graphlib::Node *n) -> bool {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(n);
        if (!op or op->op_name() != "add")
            return false;
        std::vector<Node *> operands = graph->data_operands(n);
        return std::all_of(operands.begin(), operands.end(), is_gather_op);
    };

    std::vector<graphlib::Node *> gather_ops = graph->nodes(is_gather_collapse);
    for (graphlib::Node *node : gather_ops) {
        log_trace(LogAutograd, "Found gather op: {}", node->name());
    }
}

}
