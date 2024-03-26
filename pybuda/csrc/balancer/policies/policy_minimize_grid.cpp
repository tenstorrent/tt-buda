// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_minimize_grid.hpp"

#include "balancer/balancer.hpp"
#include "utils/logger.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer {
BalancerPolicySolution run_policy_minimize_grid(Graph const* graph, BalancerConfig const&, legalizer::GraphSolver& graph_solver)
{
    for (Node* node : tt::graphlib::topological_sort(*graph)) {
        if (node->node_type() != NodeType::kBudaOp)
            continue;

        auto legal_op_models = graph_solver.at(node);
        std::vector<OpModel> op_models(legal_op_models.begin(), legal_op_models.end());
        std::sort(
            op_models.begin(),
            op_models.end(),
            [](OpModel const& a, OpModel const& b) -> bool
            {
                int perimeter_a = a.grid_shape.r + a.grid_shape.c;
                int perimeter_b = b.grid_shape.r + b.grid_shape.c;
                if (perimeter_a == perimeter_b)
                    return a.grid_shape.r < b.grid_shape.r;
                return perimeter_a < perimeter_b;
            });
        graph_solver.set(node, op_models.front());
        log_debug(LogBalancer, "Selected minimum grid for node: {}", node->name());
        log_debug(LogBalancer, "  {} {}", op_models.front().grid_shape, op_models.front().t_stream_factor);
    }

    return BalancerPolicySolution(graph_solver.finish());
}

}  // namespace tt::balancer
