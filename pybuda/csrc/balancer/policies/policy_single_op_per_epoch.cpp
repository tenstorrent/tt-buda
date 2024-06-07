// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_single_op_per_epoch.hpp"

#include <algorithm>
#include <random>

#include "balancer/policies/policy_manager.hpp"
#include "balancer/policies/policy_utils.hpp"
#include "graph_lib/node_types.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{

BalancerPolicySolution run_policy_single_op_per_epoch(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver)
{
    (void)config;
    log_debug(LogBalancer, "Starting single op per epoch balancing.");

    PolicyManager policy_manager(graph, config, graph_solver);

    // Pick OpModel for each node.
    //
    while (const graphlib::Node* node = policy_manager.get_next_op())
    {
        legalizer::GraphSolver::RemainingOpModels op_models = policy_manager.at(node);

        OpModel max_op_model = *(op_models.begin());

        for (const OpModel& op_model_it : op_models) {
            if (max_op_model.grid_shape.volume() < op_model_it.grid_shape.volume()) {
                max_op_model = op_model_it;
            }
        }

        policy_manager.commit_op(max_op_model);
        policy_manager.finish_current_epoch();

        log_debug(LogBalancer, "Selected maximum grid for node: {}", node->name());
        log_debug(LogBalancer, "  {} {}", max_op_model.grid_shape, max_op_model.t_stream_factor);
    }

    return policy_manager.commit_solution();
}

}  // namespace tt::balancer
